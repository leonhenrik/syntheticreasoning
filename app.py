import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiment import MAX_WORKERS, REQUEST_TIMEOUT, get_models, load_api_key, score


ROOT = Path(__file__).parent
MAX_TEXT_LENGTH = 200_000
MAX_REQUEST_BYTES = MAX_TEXT_LENGTH * 4 + 128
PROMPT_COLORS = {
    "supportive": "#8f9275",
    "strict": "#d27b61",
    "neutral": "#727d9b",
}
EXPECTED_ERRORS = (
    urllib.error.URLError,
    OSError,
    KeyError,
    IndexError,
    ValueError,
    RuntimeError,
    AttributeError,
    TypeError,
)
GOOGLE_API = "https://generativelanguage.googleapis.com/v1beta"


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        pages = {
            "/": "index.html",
            "/index.html": "index.html",
            "/games": "games.html",
            "/games.html": "games.html",
        }
        if self.path == "/api/models":
            self.send_models()
            return
        if self.path not in pages:
            self.send_error(404)
            return
        page = (ROOT / pages[self.path]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def send_models(self):
        api_key = ""
        try:
            api_key = load_gemini_api_key()
            models = get_google_models(api_key)
            if not models:
                self.send_json_error(502, "Google returned no models that support content generation.")
                return
            self.send_json(200, {"models": models})
        except (OSError, RuntimeError, KeyError, ValueError, AttributeError, TypeError) as error:
            message = str(error).replace(api_key, "[redacted]") if api_key else str(error)
            self.send_json_error(502, message)

    def do_POST(self):
        if self.path == "/api/game":
            self.play_game()
            return
        if self.path != "/api/evaluate":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.send_json_error(400, "Text must be between 1 character and 200,000 characters.")
                return
            payload = json.loads(self.rfile.read(length))
            writing = payload.get("text", "").strip()
            if not writing:
                self.send_json_error(400, "Paste some writing before evaluating.")
                return
            if len(writing) > MAX_TEXT_LENGTH:
                self.send_json_error(400, "Text must be 200,000 characters or fewer.")
                return
            api_key = load_api_key()
            with (ROOT / "prompts.json").open(encoding="utf-8") as file:
                prompts = json.load(file)["evaluation_prompts"]
            models = get_models(api_key)
            if not models:
                self.send_json_error(502, "NVIDIA returned no available chat-completion models.")
                return
        except (json.JSONDecodeError, AttributeError, TypeError):
            self.send_json_error(400, "Request body must contain a text string.")
            return
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            self.send_json_error(502, str(error))
            return

        requests_total = len(models) * len(prompts)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.write_event({"type": "start", "total": requests_total, "models": len(models)})
        completed = 0
        successful = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            requests = {
                pool.submit(score, model, prompt["instruction"], writing, api_key): (
                    model,
                    prompt["name"],
                )
                for model in models
                for prompt in prompts
            }
            for request in as_completed(requests):
                model, prompt_name = requests[request]
                completed += 1
                try:
                    value = request.result()
                    successful += 1
                    self.write_event(
                        {
                            "type": "score",
                            "model": model,
                            "prompt": prompt_name,
                            "score": value,
                            "completed": completed,
                            "total": requests_total,
                        }
                    )
                except EXPECTED_ERRORS as error:
                    message = str(error).replace(api_key, "[redacted]")
                    self.write_event(
                        {
                            "type": "failure",
                            "model": model,
                            "prompt": prompt_name,
                            "message": message,
                            "completed": completed,
                            "total": requests_total,
                        }
                    )

        self.write_event({"type": "done", "successful": successful, "total": requests_total})

    def send_json_error(self, status, message):
        self.send_json(status, {"error": message})

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def play_game(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                self.send_json_error(400, "The game request is invalid.")
                return
            payload = json.loads(self.rfile.read(length))
            model_a = payload.get("model_a")
            model_b = payload.get("model_b")
            rounds = payload.get("rounds")
            if (
                not isinstance(model_a, str)
                or not isinstance(model_b, str)
                or model_a == model_b
                or isinstance(rounds, bool)
                or not isinstance(rounds, int)
                or not 1 <= rounds <= 20
            ):
                self.send_json_error(400, "Choose two different models and between 1 and 20 rounds.")
                return
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            self.send_json_error(400, "Request body must contain two model IDs and a round count.")
            return

        api_key = ""
        try:
            api_key = load_gemini_api_key()
            available = {model["id"] for model in get_google_models(api_key)}
        except (OSError, RuntimeError, KeyError, ValueError, AttributeError, TypeError) as error:
            message = str(error).replace(api_key, "[redacted]") if api_key else str(error)
            self.send_json_error(502, message)
            return
        if model_a not in available or model_b not in available:
            self.send_json_error(400, "Choose models from the available Gemini model list.")
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.write_event({"type": "start", "total": rounds})
        history = []
        score_a = 0
        score_b = 0
        for round_number in range(1, rounds + 1):
            with ThreadPoolExecutor(max_workers=2) as pool:
                requests = (
                    pool.submit(game_decision, model_a, "A", round_number, rounds, history, api_key),
                    pool.submit(game_decision, model_b, "B", round_number, rounds, history, api_key),
                )
                try:
                    decision_a, decision_b = (request.result() for request in requests)
                except EXPECTED_ERRORS as error:
                    message = str(error).replace(api_key, "[redacted]")
                    self.write_event({"type": "failure", "round": round_number, "message": message})
                    return

            payoff_a, payoff_b = payoff(decision_a["move"], decision_b["move"])
            score_a += payoff_a
            score_b += payoff_b
            event = {
                "type": "round",
                "round": round_number,
                "a": decision_a,
                "b": decision_b,
                "payoff_a": payoff_a,
                "payoff_b": payoff_b,
                "score_a": score_a,
                "score_b": score_b,
            }
            self.write_event(event)
            history.append(event)
        self.write_event({"type": "done", "total": rounds})

    def write_event(self, event):
        self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
        self.wfile.flush()


def game_decision(model, player, round_number, total_rounds, history, api_key):
    previous_rounds = "\n".join(
        f"Round {event['round']}: you chose {event[player.lower()]['move']}; "
        f"your opponent chose {event['b' if player == 'A' else 'a']['move']}."
        for event in history
    ) or "No previous rounds."
    prompt = (
        "You are playing the prisoner's dilemma as Player "
        f"{player}. Both players choose simultaneously. In each round, "
        "cooperation earns 3 each if both cooperate; if you defect while "
        "the other cooperates, you earn 5 and they earn 0; mutual defection "
        "earns 1 each. Choose either cooperate or defect. This is a fixed "
        f"{total_rounds}-round match, currently round {round_number}; consider "
        "the possibility of future rounds, but make your own decision.\n\n"
        f"Previous rounds:\n{previous_rounds}\n\n"
        'Return only a JSON object with exactly these fields: '
        '{"move":"cooperate" or "defect","reason":"one concise sentence explaining your choice"}.'
    )
    result = google_api_request(
        f"{GOOGLE_API}/models/{urllib.parse.quote(model, safe='-_.')}:generateContent",
        api_key,
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 180,
                "responseMimeType": "application/json",
                "temperature": 0.7,
            },
        },
    )
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ValueError(f"{model} returned no decision candidate.")
    candidate = candidates[0]
    content = candidate.get("content")
    parts = content.get("parts", []) if isinstance(content, dict) else []
    if not isinstance(parts, list):
        raise ValueError(f"{model} returned invalid decision content.")
    response_text = "".join(
        part["text"] for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    if not response_text:
        raise ValueError(
            f"{model} returned no decision text (finish reason: {candidate.get('finishReason')!r})."
        )
    try:
        decision = json.loads(response_text.strip())
    except json.JSONDecodeError:
        start = response_text.find("{")
        if start < 0:
            raise ValueError(f"{model} did not return a JSON decision.")
        try:
            decision, _ = json.JSONDecoder().raw_decode(response_text[start:])
        except json.JSONDecodeError as error:
            raise ValueError(f"{model} returned an invalid JSON decision.") from error
    if not isinstance(decision, dict):
        raise ValueError(f"{model} did not return a JSON decision object.")
    move = decision.get("move")
    reason = decision.get("reason")
    if not isinstance(move, str) or move.strip().lower() not in ("cooperate", "defect"):
        raise ValueError(f"{model} returned an invalid move; expected cooperate or defect.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{model} returned no explanation for its decision.")
    return {"move": move.strip().lower(), "reason": reason.strip()[:500]}


def payoff(move_a, move_b):
    if move_a == "cooperate" and move_b == "cooperate":
        return 3, 3
    if move_a == "defect" and move_b == "cooperate":
        return 5, 0
    if move_a == "cooperate" and move_b == "defect":
        return 0, 5
    return 1, 1


def load_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and (ROOT / ".env.local").exists():
        for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                api_key = value.strip().strip("\"'")
                break
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env.local.")
    return api_key


def google_api_request(url, api_key, data=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 429 and attempt < 2:
                try:
                    delay = float(error.headers.get("Retry-After", 2**attempt))
                except (TypeError, ValueError):
                    delay = 2**attempt
                time.sleep(max(0, min(delay, 30)))
                continue
            raise RuntimeError(f"Google Gemini API HTTP {error.code}: {detail}") from error
    raise RuntimeError("Google Gemini API request failed after retries.")


def get_google_models(api_key):
    models = {}
    page_token = None
    seen_page_tokens = set()
    while True:
        params = {"pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        query = urllib.parse.urlencode(params)
        result = google_api_request(f"{GOOGLE_API}/models?{query}", api_key)
        if not isinstance(result, dict):
            raise ValueError("Google returned an invalid model catalog.")
        records = result.get("models", [])
        if not isinstance(records, list):
            raise ValueError("Google returned an invalid model catalog.")
        for model in records:
            if not isinstance(model, dict):
                continue
            methods = model.get("supportedGenerationMethods", [])
            name = model.get("name")
            if (
                not isinstance(name, str)
                or not isinstance(methods, list)
                or "generateContent" not in methods
            ):
                continue
            model_id = name.removeprefix("models/")
            if not model_id:
                continue
            display_name = model.get("displayName")
            models[model_id] = {
                "id": model_id,
                "name": display_name if isinstance(display_name, str) and display_name else model_id,
            }
        next_page = result.get("nextPageToken")
        if not next_page:
            break
        if not isinstance(next_page, str) or next_page in seen_page_tokens:
            raise ValueError("Google returned an invalid model catalog page token.")
        seen_page_tokens.add(next_page)
        page_token = next_page
    return sorted(models.values(), key=lambda model: (model["name"].casefold(), model["id"]))


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print(f"Synthetic Reasoning is running at http://{address[0]}:{address[1]}")
    try:
        ThreadingHTTPServer(address, AppHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
