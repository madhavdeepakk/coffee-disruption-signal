# Hosting the dashboard on Streamlit Community Cloud

The dashboard is a Streamlit app. This guide puts it online for free at a
public URL (for example `https://<name>.streamlit.app`). It hosts the
**read-only viewer**: visitors can browse the price chart, the flagged
anomalies, the precomputed explanations, the forecast, and the evaluation
tables. The buttons that make live GDELT / Gemini calls are hidden on the
public site (see step 3), so no visitor can spend your API quota.

## What you need
- A free GitHub account (https://github.com).
- A free Streamlit Community Cloud account (https://share.streamlit.io) —
  sign in with the same GitHub account.

## Step 1 — Put the project on GitHub

From the project folder (`D:\CAPSTONE`), in a terminal:

```
cd /d D:\CAPSTONE
git init
git add .
git commit -m "Coffee disruption signal dashboard"
```

Create a new **empty** repository on github.com (no README, no .gitignore —
this project already has one). Then connect and push (replace the URL with
the one GitHub shows you):

```
git remote add origin https://github.com/<your-username>/coffee-disruption-signal.git
git branch -M main
git push -u origin main
```

The `.gitignore` in this project already keeps your `.env` (API key) out of
the commit. Double-check that `.env` is **not** listed by `git status` before
you push.

## Step 2 — Create the app on Streamlit Cloud
1. Go to https://share.streamlit.io and click **New app**.
2. Choose your repository, branch `main`, and set the **Main file path** to:
   ```
   src/dashboard/app.py
   ```
3. Open **Advanced settings** and set the Python version to **3.11**.

## Step 3 — Turn on read-only mode (important for a public site)
Still in **Advanced settings**, under **Secrets**, add this line:

```
READ_ONLY = "1"
```

This hides the "Run the pipeline live" and "news signal" buttons so the public
app only shows results you generated in advance. (You do **not** need to add a
Gemini key — the public app never calls the LLM.)

## Step 4 — Deploy
Click **Deploy**. The first build takes a couple of minutes while it installs
`requirements.txt`. When it finishes you get a public URL you can share.

## Adding new explained dates later
The public site only shows explanations that already exist in `results/`. To
add more:
1. On your own machine, run the pipeline for a date:
   `python -m src.pipeline --date 2025-09-15`
2. Commit and push the new `results/pipeline_output_coffee_*.json` file:
   `git add results && git commit -m "Add explanation" && git push`
3. Streamlit Cloud redeploys automatically within a minute.

## Notes
- Keep `data/raw/*.csv` and `results/*` committed — the app reads them.
- Never commit `.env`; put secrets only in the Streamlit Cloud **Secrets** box.
- Hugging Face Spaces (https://huggingface.co/spaces) is an equivalent free
  alternative: create a **Streamlit** Space, push the same repo, and set the
  same `READ_ONLY` variable under the Space's settings/secrets.
