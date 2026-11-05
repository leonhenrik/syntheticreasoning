# Synthetic Reasoning

A small web app for exploring how different AI evaluators score the same piece of writing. Paste text into the left panel, select **Evaluate writing**, and watch the score distributions build as results arrive.

## Web app

Set `NVIDIA_API_KEY` or `nvidia_api_key` in the environment or `.env.local` using an NVIDIA API key, install the Python dependencies, and start the local server:

```powershell
pip install -r requirements.txt
pnpm dev
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). By default, the app evaluates with `mistralai/mistral-large-4-1t-a52b`. The NVIDIA model catalog includes IDs that may not be callable for your account, so the app does not automatically send requests to every listed model. To try other catalog IDs, set `NVIDIA_MODELS` to a comma-separated list of model IDs that work with your key. The server listens on localhost and streams each completed score to the chart.

### Games

Open [http://127.0.0.1:8000/games](http://127.0.0.1:8000/games) to have two Google Gemini models play repeated rounds of the prisoner's dilemma. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment or `.env.local`, then select two different models and a round count. The page streams each simultaneous choice, the model's short explanation, cumulative scores, and cooperation rate. The payoff matrix is shown on the page. Models are loaded from Google's Gemini API and filtered for content-generation support; availability and quotas depend on your Google AI Studio project.

## Command-line evaluation

Replace `text_to_evaluate` in `prompts.json` with your writing. The three prompts ask each configured NVIDIA model for one integer score from 1 to 100: supportive, strict, and neutral.

Set `NVIDIA_API_KEY` or `nvidia_api_key` in the environment or `.env.local`, install dependencies, and run:

```powershell
pip install -r requirements.txt
python experiment.py
```

Each run saves `scores.csv` and four score-distribution graphs (total, supportive, strict, and neutral) in a new folder under `results/`. By default, it uses `mistralai/mistral-large-4-1t-a52b`; set `NVIDIA_MODELS` to a comma-separated list of model IDs to use other models. The script confirms configured IDs appear in NVIDIA's catalog, then runs up to two requests at once and retries rate-limited requests up to two times. Models that still fail are reported and skipped while successful scores are saved.
