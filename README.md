# Synthetic Reasoning

A small web app for exploring how different AI evaluators score the same piece of writing. Paste text into the left panel, select **Evaluate writing**, and watch the score distributions build as results arrive.

## Web app

Set `GEMINI_API_KEY` or `gemini_api_key` in the environment or `.env.local`, install the Python dependencies, and start the local server:

```powershell
pip install -r requirements.txt
pnpm dev
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The app compares supportive, strict, and neutral evaluations across available Gemini models. The server listens on localhost and streams each completed score to the chart.

## Command-line evaluation

Replace `text_to_evaluate` in `prompts.json` with your writing. The three prompts ask each available Gemini model for one integer score from 1 to 100: supportive, strict, and neutral.

Set `GEMINI_API_KEY` or `gemini_api_key` in the environment or `.env.local`, install dependencies, and run:

```powershell
pip install -r requirements.txt
python experiment.py
```

Each run saves `scores.csv` and four score-distribution graphs (total, supportive, strict, and neutral) in a new folder under `results/`. The script skips retired Gemini 2.5 models and image/audio-only models, and runs up to two requests at once to reduce rate-limit errors. It retries rate-limited requests up to two times. Models that still fail are reported and skipped while successful scores are saved.
