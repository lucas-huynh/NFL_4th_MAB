import json, joblib, numpy as np, pandas as pd
from pathlib import Path

ART = Path("artifacts")

# load shared artifcacts
META = json.load(open(ART / "metadata.json"))
ACTIONS = META["actions"]                              # order must match training
NUMERIC_FEATURES = META["numeric_features"]
CATEGORICAL_FEATURES = META["categorical_features"]
FEATURE_COLS = META["feature_cols"]

PRE = joblib.load(ART / "preprocessor.joblib")
ARM_EPA = joblib.load(ART / "arm_models_epa.joblib")
ARM_WPA = joblib.load(ART / "arm_models_wpa.joblib")

def _kick_distance_yards(yardline_100: float) -> float:
    # distance to posts + 17-yards snap/placement
    return float(yardline_100) + 17.0

def _fg_max_range(context: dict) -> float:
    """
    Simple context-aware cap. Start at 65 yards. 
    +3 if dome, -3 if strong wind (>=15 mph) and outdoors, -2 if very cold (<=20F) and outdoors.
    Keep conservative to avoid bogus FGs.
    """
    roof = str(context.get("roof", "outdoors")).lower()
    wind = float(context.get("wind", 0) or 0)
    temp = float(context.get("temp", 60) or 60)

    max_fg = 65.0
    if roof == "dome":
        max_fg += 3.0
    elif roof == "outdoors":
        if wind >= 15:
            max_fg -= 3.0
        if temp <= 20:
            max_fg -= 2.0
    # clamp to sane band
    return max(55.0, min(max_fg, 70.0))

# --- Punt feasibility (strategic, not physical) ---
def _punt_infeasible(context: dict) -> tuple[bool, str]:
    """
    Simple rules that reflect common analytics practice.
    Returns (infeasible_flag, reason_text).

    R1: Inside opp 35 -> almost always go/FG (punting is dominated).
    R2: Q4 late (<= 5:00), trailing or tied, at midfield+ with <= 5 to go -> don't punt.
    """
    yd100 = float(context.get("yardline_100", 99) or 99)    # 0 = opp goal line
    qtr   = int(context.get("qtr", 1) or 1)
    gsr   = int(context.get("game_seconds_remaining", 3600) or 3600)
    sd    = float(context.get("score_differential", 0) or 0)
    ytg   = float(context.get("ydstogo", 10) or 10)

    # R1: Opponent 35 or closer
    if yd100 <= 35:
        return True, "Inside opponent 35 (punting dominated by go/FG)."

    # R2: Late-game, non-leading, reasonable distance, at or beyond midfield (opp side)
    if (qtr == 4 and gsr <= 300) and (sd <= 0) and (yd100 <= 50) and (ytg <= 5):
        return True, "Late Q4, non-leading with <=5 to go on opp side (analytics: do not punt)."

    return False, ""


def _apply_action_constraints(context: dict, mu_epa, mu_wpa, actions):
    """
    In-place masks on MU arrays for impossible/implausible actions.
    Currently: mask FG attempts beyond max range.
    """
    if "fg" in actions:
        j_fg = actions.index("fg")
        yd100 = float(context.get("yardline_100", 99) or 99)
        kick_dist = _kick_distance_yards(yd100)
        max_fg = _fg_max_range(context)
        if kick_dist > max_fg:
            # make fg unpickable
            mu_epa[:, j_fg] = -1e9
            mu_wpa[:, j_fg] = -1e9

    if "punt" in actions:
        j_p = actions.index("punt")
        bad, _ = _punt_infeasible(context)
        if bad:
            mu_epa[:, j_p] = -1e9
            mu_wpa[:, j_p] = -1e9


def _predict_per_arm(Xd, arm_models):
    mu = np.zeros((Xd.shape[0], len(ACTIONS)), dtype=float)
    for j, a in enumerate(ACTIONS):
        m = arm_models[a]
        if isinstance(m, tuple) and m[0] == "const":
            mu[:, j] = m[1]
        else:
            mu[:, j] = m.predict(Xd)
    return mu

def _to_df(context_dict: dict) -> pd.DataFrame:
    # ensure all expected columns exist; preprocessor will impute/encode.
    row = {c: context_dict.get(c, None) for c in FEATURE_COLS}
    # optional: normalize a few categorical inputs
    for key in ["posteam","defteam","home_team","away_team","posteam_type","roof","surface"]:
        if key in row and isinstance(row[key], str):
            row[key] = row[key].strip()
    return pd.DataFrame([row])

def score_context(context: dict, metric: str = "wpa"):
    """
    Score one context and return:
      epa_scores: dict(action -> μ̂_EPA)
      wpa_scores: dict(action -> μ̂_WPA)
      recommended_action: str (argmax of chosen metric)
      details: {"epa": epa_scores, "wpa": wpa_scores}
    """
    df1 = _to_df(context)
    Xd = PRE.transform(df1)

    MU_epa = _predict_per_arm(Xd, ARM_EPA)   # shape (1,K)
    MU_wpa = _predict_per_arm(Xd, ARM_WPA)   # shape (1,K)

    _apply_action_constraints(context, MU_epa, MU_wpa, ACTIONS)

    pick = MU_wpa if metric.lower() == "wpa" else MU_epa
    j = int(np.argmax(pick[0]))
    rec = ACTIONS[j]

    epa_scores = {a: float(MU_epa[0, i]) for i, a in enumerate(ACTIONS)}
    wpa_scores = {a: float(MU_wpa[0, i]) for i, a in enumerate(ACTIONS)}
    return epa_scores, wpa_scores, rec, {"epa": epa_scores, "wpa": wpa_scores}
