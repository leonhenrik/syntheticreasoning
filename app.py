import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiment import MAX_WORKERS, get_models, load_api_key, score


ROOT = Path(__file__).parent
MAX_TEXT_LENGTH = 200_000
MAX_REQUEST_BYTES = MAX_TEXT_LENGTH * 4 + 128
PROMPT_COLORS = {
    "supportive": "#8f9275",
    "strict": "#d27b61",
    "neutral": "#727d9b",
}
EXPECTED_ERRORS = (urllib.error.URLError, OSError, KeyError, IndexError, ValueError, RuntimeError)


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        page = (ROOT / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
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
                self.send_json_error(502, "Gemini returned no models that support writing evaluation.")
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
                            "model": model[7:] if model.startswith("models/") else model,
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
                            "model": model[7:] if model.startswith("models/") else model,
                            "prompt": prompt_name,
                            "message": message,
                            "completed": completed,
                            "total": requests_total,
                        }
                    )

        self.write_event({"type": "done", "successful": successful, "total": requests_total})

    def send_json_error(self, status, message):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_event(self, event):
        self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
        self.wfile.flush()


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print(f"Synthetic Reasoning is running at http://{address[0]}:{address[1]}")
    try:
        ThreadingHTTPServer(address, AppHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
