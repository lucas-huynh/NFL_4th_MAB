# 4th-Down Decision Calculator

A reproducible pipeline + Streamlit app for **NFL 4th-down decisions**.  
It builds recent team features from play-by-play, trains a **behavior policy** (multinomial logistic regression), fits per-action (arm) models for **EPA/WPA**, and serves **Greedy** and **LinUCB** multi-armed bandit recommendations with friendly explanations.

**Python 3.9–3.12 recommended.**

---

## Repo Layout

```text
.
├── app.py                         # Streamlit UI
├── artifacts/                     # Trained models + inference code
│   ├── inference.py               # score_context(), ACTIONS, preprocessor, etc.
│   ├── test_infer.py              # quick tests for inference flow
│   ├── arm_models_epa.joblib      # per-action regressors (EPA)
│   ├── arm_models_wpa.joblib      # per-action regressors (WPA)
│   └── META / *.json / *.joblib   # metadata, ColumnTransformer, encoders, etc.
├── data/                          # CSVs produced by data_clean
│   ├── pbp_clean_2016_2024.csv
│   ├── decisions_2016_2024.csv
│   └── (other 4th-down/metrics/situational CSVs)
├── team_logos/                    # PNG logos for the app (e.g., KC.png)
├── NFL.png                        # header logo for UI
├── field_diagram.png              # yardline helper image
├── data_clean_2016_2024.ipynb     # pulls PBP via nfl_data_py (nflverse) + feature build
├── behavior_2016_2024_epa.ipynb   # behavior policy + Greedy/LinUCB + EPA arm models
├── behavior_2016_2024_wpa.ipynb   # behavior policy + Greedy/LinUCB + WPA arm models
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip

# core deps
pip install streamlit pandas numpy scikit-learn scipy joblib matplotlib
pip install nfl_data_py        # pulls play-by-play from nflverse
```

---

## Workflow (End-to-End)

### 1) Data clean & feature build → `data_clean_2016_2024.ipynb`

- Pulls play-by-play from **nfl_data_py** (nflverse).
- Builds features:
  - Offense/Defense EPA 4-week rolling (`off_epa_4w`, `def_epa_4w`)
  - FG accuracy by distance bands (short/mid/long) with a shifted 16-game lookback
  - Punt net (4-week) `punt_net_4w` using snap yardline & next receiving snap  
    ```
    net = yardline_100 at punt + yardline_100 on next receiving snap − 100
    ```
    Then apply a team-shifted 4-week rolling mean.
  - Drive/fatigue context (`plays_in_drive_so_far`, `def_time_on_field_cum/share`, etc.)

- Writes:
  - `data/pbp_clean_2016_2024.csv`
  - `data/decisions_2016_2024.csv`

---

### 2) Behavior policy & arm models

Run both notebooks:

- `behavior_2016_2024_epa.ipynb`
- `behavior_2016_2024_wpa.ipynb`

These will:

- Train a multinomial **LogisticRegression** behavior policy \( P_b(a \mid x) \) and report accuracy.
- Fit per-action **Ridge** regressors (arm models) for **EPA** or **WPA**.
- Evaluate **Greedy** (argmax μ̂) and **LinUCB** (plus ε-greedy) with Off-Policy Evaluation (**DR/IPS/ESS** + bootstrap CIs).
- Dump artifacts to `artifacts/`:
  - `arm_models_epa.joblib`, `arm_models_wpa.joblib`
  - Preprocessor + META **JSONs/joblibs**

## Bandit Formulas

- **Greedy Policy**  
  \[
  a_t = \arg\max_a \hat{\mu}(x_t, a)
  \]

- **ε-Greedy Policy**  
  \[
  a_t =
  \begin{cases}
  \arg\max_a \hat{\mu}(x_t, a) & \text{with probability } 1-\epsilon \\
  \text{random action} & \text{with probability } \epsilon
  \end{cases}
  \]

- **LinUCB (per-action confidence bound)**  
  \[
  a_t = \arg\max_a \Big( \hat{\mu}(x_t, a) +
  \alpha \sqrt{x_t^\top A_a^{-1} x_t} \Big)
  \]
  where:
  - \( A_a \) is the regularized design matrix for arm \( a \)  
  - \( \hat{\mu}(x_t, a) \) is the predicted reward  
  - \( \alpha \) tunes exploration

---

### 3) Sanity-check inference (optional)

```bash
python artifacts/test_infer.py
# or
python -m artifacts.test_infer
```

---

### 4) Run the app

```bash
streamlit run app.py
```

The app loads artifacts from `artifacts/` and team-week aggregates from `data/decisions_2016_2024.csv`.

---

## App Features (`app.py`)

- **Teams & possession pickers** with team logos from `team_logos/`.
- **Stadium auto-mapping**: home team → roof/surface defaults (editable; stadium name shown).
- **Venue & weather**: dome-aware defaults for temperature/wind.
- **Situation inputs**: quarter/time, yardline helper diagram, yards-to-go, scores, timeouts.
- **Auto-filled recent metrics** (from `decisions_2016_2024.csv`):
  - `off_epa_4w`, `def_epa_4w`
  - FG% short/mid/long
  - Punt net yards (4-week avg)
- **Recommendation box**:
  - Optimize for **WPA** or **EPA**
  - Shows **Greedy** pick with deltas vs alternatives
  - Flags **infeasible** actions

---

## What the Folders Contain

- **`data_clean_2016_2024.ipynb`** — pulls PBP via `nfl_data_py`, builds features; writes CSVs under `data/`.
- **`behavior_*.ipynb`** — run Greedy & LinUCB evaluations, train the behavior policy (logistic regression), and fit/store arm models for EPA/WPA.
- **`artifacts/`** — model artifacts & code used by the app:
  - `inference.py` (exports `score_context()`)
  - `test_infer.py`
  - `arm_models_epa.joblib`, `arm_models_wpa.joblib`
  - Preprocessor/META joblibs/JSONs
- **`data/`** — CSVs for all 4th-down plays, decisions, advanced metrics, situational info, etc.
- **`team_logos/`** — PNGs (named by team abbr, e.g., `KC.png`) for the Streamlit UI.

---

## Punt Rationale

- **Punt net (4w)** uses final field position instead of raw kick/return fields, so returns, touchbacks, OB, and most penalties are naturally included:  
  ```
  net = S + N − 100
  ```
  where `S` = yardline_100 at punt, and `N` = yardline_100 on the next snap by the receiving team.  
  Then take a **shifted 4-week rolling mean** per team to capture recent special-teams performance.

- **Predict-then-Optimize**: arm models predict outcome value **μ̂(x, a)** for each action; the policy (**Greedy** or **LinUCB**) optimizes over these predictions to pick the action.

---

## Minimal Requirements (`requirements.txt`)

```
streamlit
pandas
numpy
scikit-learn
scipy
joblib
matplotlib
nfl_data_py
```

Install with:

```bash
pip install -r requirements.txt
```


