4th-Down Decision Calculator

A reproducible pipeline + Streamlit app for NFL 4th-down decisions.
It builds recent team features from play-by-play, trains a behavior policy (multinomial logistic regression), fits per-action (arm) models for EPA/WPA, and serves Greedy and LinUCB multi-armed bandit recommendations with friendly explanations.

Python 3.9–3.12 recommended.

Repo Layout
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
├── team_logos/                    # PNG logos for the app (file name = team abbr, e.g., KC.png)
├── NFL.png                        # header logo for UI
├── field_diagram.png              # yardline helper image
├── data_clean_2016_2024.ipynb     # pulls PBP via nfl_data_py (nflverse) + feature build
├── behavior_2016_2024_epa.ipynb   # behavior policy + Greedy/LinUCB + EPA arm models
├── behavior_2016_2024_wpa.ipynb   # behavior policy + Greedy/LinUCB + WPA arm models
└── README.md

Installation
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip

# core deps
pip install streamlit pandas numpy scikit-learn scipy joblib matplotlib
pip install nfl_data_py        # pulls play-by-play from nflverse

Workflow (end-to-end)

Typical order I run things:

Data clean & feature build → data_clean_2016_2024.ipynb

Pulls play-by-play from nfl_data_py (nflverse).

Builds features:

Offense/Defense EPA 4-week rolling (off_epa_4w, def_epa_4w)

FG accuracy by distance bands (short/mid/long) with a shifted 16-game lookback

Punt net (4-week) punt_net_4w using snap yardline & next receiving snap
(net = yardline_100 at punt + yardline_100 on next receiving snap − 100; then shift+roll(4))

Drive/fatigue context (plays_in_drive_so_far, def_time_on_field_cum/share, etc.)

Writes:

data/pbp_clean_2016_2024.csv

data/decisions_2016_2024.csv

Behavior policy & arm models
Run both notebooks:

behavior_2016_2024_epa.ipynb

behavior_2016_2024_wpa.ipynb
These:

Train a multinomial LogisticRegression behavior policy 
𝑃
𝑏
(
𝑎
∣
𝑥
)
P
b
	​

(a∣x) and report accuracy.

Fit per-action Ridge regressors (arm models) for EPA or WPA.

Evaluate Greedy (argmax μ̂) and LinUCB (plus ε-greedy) using Off-Policy Evaluation (DR/IPS/ESS + bootstrap CIs).

Dump artifacts to artifacts/:

arm_models_epa.joblib, arm_models_wpa.joblib

preprocessor + META JSONs/joblibs.

Sanity-check inference (optional)

python artifacts/test_infer.py
# or
python -m artifacts.test_infer


Run the app

streamlit run app.py


The app loads artifacts from artifacts/ and team-week aggregates from data/decisions_2016_2024.csv.

App Features (app.py)

Teams & possession pickers with team logos from team_logos/.

Stadium auto-mapping: home team → roof/surface defaults (editable; stadium name shown).

Venue & weather (dome-aware defaults for temp/wind).

Situation inputs: quarter/time, yardline helper diagram, yards-to-go, scores, timeouts.

Auto-filled recent metrics (from decisions_2016_2024.csv):

off_epa_4w, def_epa_4w

FG% short/mid/long

Punt net yards (4-week avg)

Recommendation box:

Optimize for WPA or EPA

Shows Greedy pick with deltas vs alternatives; flags infeasible actions.

What the folders contain

data_clean_2016_2024.ipynb — pulls PBP via nfl_data_py and builds features; writes CSVs under data/.

behavior_*.ipynb — run Greedy & LinUCB evaluations, train the behavior policy (logistic regression), and fit/store arm models for EPA/WPA.

artifacts/ — model artifacts & code used by the app:

inference.py (exports score_context()), test_infer.py, arm_models_epa.joblib, arm_models_wpa.joblib, preprocessor/META joblibs/JSONs.

data/ — CSVs for all 4th-down plays, decisions, advanced metrics, situational info, etc.

team_logos/ — PNGs (named by team abbr, e.g., KC.png) for the Streamlit UI.

Rationale (quick)

Punt net (4w) uses final field position instead of raw kick/return fields, so returns, touchbacks, OB, and most penalties are naturally included:

net
=
𝑆
+
𝑁
−
100
net=S+N−100

where 
𝑆
S = yardline_100 at punt, 
𝑁
N = yardline_100 on the next snap by the receiving team.
Then take a shifted 4-week rolling mean per team to capture recent special-teams performance.

Predict-then-Optimize: arm models predict outcome value 
𝜇
^
(
𝑥
,
𝑎
)
μ
^
	​

(x,a) for each action; the policy (Greedy or LinUCB) optimizes over these predictions to pick the action.

Troubleshooting

Logos/images not showing → ensure NFL.png, field_diagram.png, and team_logos/ABBR.png exist at the paths used in app.py.

Artifacts missing → re-run the behavior notebooks to create arm_models_*.joblib and the preprocessor/META joblibs.

Duplicate widget IDs → if you add widgets, give them unique key= values.

Path issues → the app expects paths relative to the repo root.

