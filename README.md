---
title: Corrosion PdM
emoji: 🛢️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# Pipeline Corrosion

Predicts pipeline **corrosion rate**, **thickness-loss rate**, **NACE risk level**,
**remaining useful life (RUL)** and an **intervention-priority score** from operating
and fluid-chemistry inputs. Supports a manual form or batch file upload, with SHAP
explanations and an LSTM-based RUL forecast tab.

The trained models (Random Forest / XGBoost / LSTM) ship in `models/`, so the app
runs without retraining.

## Run locally

Requires **Python 3.12**.

```bash
# from inside the corrosion-pdm folder
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at http://localhost:8501. The trained models are bundled, so no
training is needed. Uploaded datasets can use English, French or abbreviated
column names — the **Batch upload** tab auto-maps them and lets you confirm the
mapping before scoring.

If a model fails to load (package-version mismatch on another machine), rebuild
it with `python train.py` then `python lstm_rul.py`.

## Accounts & data

The app requires a **sign-up / login** — passwords are salted and PBKDF2-hashed,
never stored in clear text.

The **Database** tab ships with an existing reference dataset of **10,000 corrosion
records** (`data/raw/corrosion_10000.xlsx`), shown out of the box. Logged-in users
can **optionally add their own** data (from the Single segment or Batch upload tabs):
each scored row is **appended** on top — it never replaces the built-in data — and is
**private** to that account. User additions live in a local SQLite file (`app_data.db`,
created on first run, not shipped). You can view the reference set, your own additions,
or the combined database, and download any of them.
