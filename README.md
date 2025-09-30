# 4th-Down Decision Calculator

A reproducible pipeline + Streamlit app for **NFL 4th-down decisions**.  
It builds recent team features from play-by-play, trains a **behavior policy** (multinomial logistic regression), fits **per-action (arm) models** for EPA/WPA, and serves **Greedy** and **LinUCB** multi-armed bandit recommendations with friendly explanations.

---

## Repo Layout
.
├── app.py                       # Streamlit UI
├── artifacts/                   # Trained models + inference code
│   ├── inference.py             # score_context(), ACTIONS, preprocessor, etc.
│   ├── test_infer.py            # quick tests for inference
│   ├── arm_models_epa.joblib    # per-action regressors (EPA)
│   ├── arm_models_wpa.joblib    # per-action regressors (WPA)
│   ├── *.json / *.joblib        # metadata (META), ColumnTransformer, encoders, etc.
├── data/                        # CSVs produced by data_clean
│   ├── pbp_clean_2016_2024.csv
│   ├── decisions_2016_2024.csv
│   └── (other 4th-down/metrics/situational CSVs)
├── team_logos/                  # PNG logos used by the app (names = team abbr e.g., KC.png)
├── NFL.png                      # small header logo for UI
├── field_diagram.png            # yardline helper image
├── data_clean_2016_2024.ipynb   # build features from nfl_data_py (nflverse)
├── behavior_2016_2024_epa.ipynb # behavior policy + arm models (EPA)
├── behavior_2016_2024_wpa.ipynb # behavior policy + arm models (WPA)
└── README.md

## Quick Start

### 1) Create environment & install deps

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -U pip

# core deps
pip install streamlit pandas numpy scikit-learn scipy joblib matplotlib
pip install nfl_data_py           # pulls play-by-play from nflverse

Python 3.9–3.12 recommended.


