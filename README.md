# Gemini writing evaluation

Replace `text_to_evaluate` in `prompts.json` with your writing. The three prompts ask each available Gemini model for one integer score from 1 to 100: supportive, strict, and neutral.

Set `GEMINI_API_KEY` or `gemini_api_key` in the environment or `.env.local`, install dependencies, and run:

```powershell
pip install -r requirements.txt
python experiment.py
```

Each run saves `scores.csv` and four score-distribution graphs (total, supportive, strict, and neutral) in a new folder under `results/`. The script skips retired Gemini 2.5 models and image/audio-only models, and runs up to two requests at once to reduce rate-limit errors. It retries rate-limited requests up to two times. Models that still fail are reported and skipped while successful scores are saved.
