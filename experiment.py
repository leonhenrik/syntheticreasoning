import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


API = "https://generativelanguage.googleapis.com/v1beta"
MAX_WORKERS = 2
def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
    if not api_key and Path(".env.local").exists():
        for line in Path(".env.local").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in ("GEMINI_API_KEY", "gemini_api_key"):
                api_key = value.strip().strip("\"'")
                break
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or gemini_api_key in your environment or .env.local.")
    return api_key


def api_request(url, data=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 429 and attempt < 2:
                try:
                    delay = float(error.headers.get("Retry-After", 2**attempt))
                except (TypeError, ValueError):
                    delay = 2**attempt
                time.sleep(min(delay, 30))
                continue
            raise RuntimeError(f"Gemini API HTTP {error.code}: {detail}") from error


def get_models(api_key):
    models = []
    page_token = ""
    while True:
        query = urllib.parse.urlencode({"key": api_key, "pageSize": 1000, "pageToken": page_token})
        page = api_request(f"{API}/models?{query}")
        models.extend(
            model_name
            for model in page.get("models", [])
            if "generateContent" in model.get("supportedGenerationMethods", [])
            and (model_name := model.get("name", "")).startswith("models/")
            and "2.5" not in model_name
            and not any(term in model_name.lower() for term in ("image", "tts", "banana"))
        )
        page_token = page.get("nextPageToken", "")
        if not page_token:
            return models


def score(model, instruction, writing, api_key):
    prompt = (
        f"{instruction}\nGive the writing a score from 1 to 100. "
        "Return only an integer.\n\nWriting:\n"
        f"{writing}"
    )
    query = urllib.parse.urlencode({"key": api_key})
    result = api_request(
        f"{API}/{model}:generateContent?{query}",
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
    )
    value = json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
    if isinstance(value, dict):
        value = value.get("score")
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError(f"Expected an integer from 1 to 100; received {value!r}")
    return value


def plot_scores(name, scores, output):
    counts = Counter(scores)
    plt.figure()
    plt.plot(range(1, 101), [counts[score] for score in range(1, 101)])
    plt.xlim(1, 100)
    plt.xlabel("Score")
    plt.ylabel("Number of responses")
    plt.title(f"{name} score distribution")
    plt.savefig(output / f"{name.lower()}.png")
    plt.close()


def main():
    try:
        with open("prompts.json", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"prompts.json is invalid at line {error.lineno}, column {error.colno}. "
            "For multi-line writing, escape line breaks as \\n inside the JSON string."
        ) from error

    try:
        api_key = load_api_key()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    writing = config["text_to_evaluate"]
    prompts = config["evaluation_prompts"]
    models = get_models(api_key)
    if not models:
        raise SystemExit("Gemini returned no models that support generateContent.")

    output = Path("results") / datetime.now().strftime("%Y%m%dT%H%M%S")
    output.mkdir(parents=True)
    scores_by_prompt = {prompt["name"]: [] for prompt in prompts}
    all_scores = []

    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        rows = csv.writer(file)
        rows.writerow(["model", "prompt", "score"])
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            requests = {
                pool.submit(score, model, prompt["instruction"], writing, api_key): (model, prompt["name"])
                for model in models
                for prompt in prompts
            }
            for request in as_completed(requests):
                model, prompt_name = requests[request]
                try:
                    value = request.result()
                    rows.writerow([model, prompt_name, value])
                    file.flush()
                    scores_by_prompt[prompt_name].append(value)
                    all_scores.append(value)
                    print(f"{model} | {prompt_name}: {value}")
                except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError, RuntimeError) as error:
                    print(f"{model} | {prompt_name}: FAILED - {str(error).replace(api_key, '[redacted]')}")

    if not all_scores:
        raise SystemExit(f"No scores received. Check {output / 'scores.csv'}.")

    plot_scores("total", all_scores, output)
    for name, scores in scores_by_prompt.items():
        if scores:
            plot_scores(name, scores, output)

    print(f"Results saved in {output}")


if __name__ == "__main__":
    main()
