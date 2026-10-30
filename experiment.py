import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


API = "https://integrate.api.nvidia.com/v1"
MAX_WORKERS = 2
REQUEST_TIMEOUT = 180
DEFAULT_MODELS = ("mistralai/mistral-large-4-1t-a52b",)


def load_api_key():
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("nvidia_api_key")
    if not api_key and Path(".env.local").exists():
        for line in Path(".env.local").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in ("NVIDIA_API_KEY", "nvidia_api_key"):
                api_key = value.strip().strip("\"'")
                break
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY or nvidia_api_key in your environment or .env.local.")
    return api_key


def api_request(url, api_key, data=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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
                time.sleep(min(delay, 30))
                continue
            raise RuntimeError(f"NVIDIA API HTTP {error.code}: {detail}") from error


def get_models(api_key):
    result = api_request(f"{API}/models", api_key)
    catalog = {
        model_id
        for model in result.get("data", [])
        if isinstance(model, dict)
        and isinstance((model_id := model.get("id")), str)
        and model_id
    }
    configured_models = os.getenv("NVIDIA_MODELS")
    if configured_models is None:
        requested_models = list(DEFAULT_MODELS)
    else:
        requested_models = list(dict.fromkeys(
            model.strip() for model in configured_models.split(",") if model.strip()
        ))
        if not requested_models:
            raise RuntimeError("NVIDIA_MODELS must contain one or more comma-separated model IDs.")

    unavailable_models = [model for model in requested_models if model not in catalog]
    if unavailable_models:
        raise RuntimeError(
            "These NVIDIA_MODELS IDs were not returned by NVIDIA's model catalog: "
            + ", ".join(unavailable_models)
        )
    return requested_models


def score(model, instruction, writing, api_key):
    prompt = (
        f"{instruction}\nGive the writing a score from 1 to 100. "
        "Return only an integer.\n\nWriting:\n"
        f"{writing}"
    )
    result = api_request(
        f"{API}/chat/completions",
        api_key,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0,
            "stream": False,
            "reasoning_effort": "high",
        },
    )
    choice = result["choices"][0]
    response_text = choice["message"]["content"]
    if not isinstance(response_text, str):
        raise ValueError(
            "NVIDIA returned no text score "
            f"(finish reason: {choice.get('finish_reason')!r})."
        )
    response_text = response_text.strip()
    value = int(response_text)
    if not 1 <= value <= 100:
        raise ValueError(f"Expected an integer from 1 to 100; received {value}")
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
        raise SystemExit("NVIDIA returned no available chat-completion models.")

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
