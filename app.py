import streamlit as st
import pandas as pd
from pathlib import Path
from artifacts.inference import score_context, ACTIONS
from artifacts.inference import META  # to access FEATURE_COLS
FEATURE_COLS = META["feature_cols"]

TEAM_ABBRS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
    "PHI","PIT","SEA","SF","TB","TEN","WAS"
]

st.set_page_config(page_title="4th-Down Decision Calculator", page_icon="🏈", layout="centered")
APP_DIR = Path(__file__).parent
logo_path = APP_DIR / "NFL.png"

# display NFL logo at the top, centered
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    st.image(str(logo_path), width=150)

# page title
st.title("4th-Down Decision Calculator")

# load decisions CSV once and pre-aggregate team-week features
@st.cache_data(show_spinner=False)
def load_team_week_features(path="data/decisions_2016_2024.csv") -> pd.DataFrame:
    usecols = ["season","week","posteam","off_epa_4w","def_epa_4w",
               "fg_pct_short","fg_pct_mid","fg_pct_long","punt_net_4w"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"]   = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["posteam"] = df["posteam"].astype(str).str.strip()

    feats = (df.groupby(["season","week","posteam"], dropna=False)
               [["off_epa_4w","def_epa_4w","fg_pct_short","fg_pct_mid","fg_pct_long","punt_net_4w"]]
               .mean().reset_index())

    feats = feats.fillna({
        "off_epa_4w": 0.0, "def_epa_4w": 0.0,
        "fg_pct_short": 0.88, "fg_pct_mid": 0.75, "fg_pct_long": 0.55, # historical averages if unknown
        "punt_net_4w": 42.0
    })
    return feats

TEAM_FEATS = load_team_week_features()

def get_team_feats(season:int, week:int, team:str):
    row = TEAM_FEATS[
        (TEAM_FEATS["season"] == season) &
        (TEAM_FEATS["week"] == week) &
        (TEAM_FEATS["posteam"].str.upper() == team.upper())
    ]
    if row.empty:
        return dict(off_epa_4w=0.0, def_epa_4w=0.0,
                    fg_pct_short=0.88, fg_pct_mid=0.75, fg_pct_long=0.55,
                    punt_net_4w=42.0)
    r = row.iloc[0]
    return dict(off_epa_4w=float(r.off_epa_4w), def_epa_4w=float(r.def_epa_4w),
                fg_pct_short=float(r.fg_pct_short), fg_pct_mid=float(r.fg_pct_mid),
                fg_pct_long=float(r.fg_pct_long), punt_net_4w=float(r.punt_net_4w))

# venue map (roof/surface) by home team
VENUE = {
    # roof: outdoors | dome | open ; surface: grass | turf
    "ARI": ("dome", "turf"),
    "ATL": ("dome", "turf"),
    "BAL": ("outdoors", "turf"),
    "BUF": ("outdoors", "turf"),
    "CAR": ("outdoors", "turf"),
    "CHI": ("outdoors", "grass"),
    "CIN": ("outdoors", "turf"),
    "CLE": ("outdoors", "turf"),
    "DAL": ("dome", "turf"),
    "DEN": ("outdoors", "turf"),
    "DET": ("dome", "turf"),
    "GB":  ("outdoors", "grass"),
    "HOU": ("dome", "turf"),
    "IND": ("dome", "turf"),
    "JAX": ("outdoors", "turf"),
    "KC":  ("outdoors", "grass"),
    "LAC": ("dome", "turf"),
    "LAR": ("dome", "turf"),
    "LV":  ("dome", "turf"),
    "MIA": ("outdoors", "grass"),
    "MIN": ("dome", "turf"),
    "NE":  ("outdoors", "turf"),
    "NO":  ("dome", "turf"),
    "NYG": ("outdoors", "turf"),
    "NYJ": ("outdoors", "turf"),
    "PHI": ("outdoors", "grass"),
    "PIT": ("outdoors", "turf"),
    "SEA": ("open", "turf"),
    "SF":  ("outdoors", "grass"),
    "TB":  ("outdoors", "grass"),
    "TEN": ("outdoors", "turf"),
    "WAS": ("outdoors", "grass"),
}

def venue_defaults(home_team:str):
    ht = (home_team or "").upper().strip()
    return VENUE.get(ht, ("outdoors", "grass"))

# for team logos
LOGO_DIR = Path(__file__).parent / "team_logos"

def team_logo(team: str):
    """Return path to team logo if found, else None."""
    path = LOGO_DIR / f"{team}.png"
    return str(path) if path.exists() else None

# team full names
TEAM_FULL_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",
    "LV":  "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF":  "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

# stadium map (name + defaults)
STADIUMS = {
    "ARI": {"name": "State Farm Stadium",               "roof": "dome",      "surface": "turf"},
    "ATL": {"name": "Mercedes-Benz Stadium",            "roof": "dome",      "surface": "turf"},
    "BAL": {"name": "M&T Bank Stadium",                 "roof": "outdoors",  "surface": "turf"},
    "BUF": {"name": "Highmark Stadium",                 "roof": "outdoors",  "surface": "turf"},
    "CAR": {"name": "Bank of America Stadium",          "roof": "outdoors",  "surface": "turf"},
    "CHI": {"name": "Soldier Field",                    "roof": "outdoors",  "surface": "grass"},
    "CIN": {"name": "Paycor Stadium",                   "roof": "outdoors",  "surface": "turf"},
    "CLE": {"name": "Cleveland Browns Stadium",         "roof": "outdoors",  "surface": "turf"},
    "DAL": {"name": "AT&T Stadium",                     "roof": "dome",      "surface": "turf"},
    "DEN": {"name": "Empower Field at Mile High",       "roof": "outdoors",  "surface": "turf"},
    "DET": {"name": "Ford Field",                       "roof": "dome",      "surface": "turf"},
    "GB":  {"name": "Lambeau Field",                    "roof": "outdoors",  "surface": "grass"},
    "HOU": {"name": "NRG Stadium",                      "roof": "dome",      "surface": "turf"},
    "IND": {"name": "Lucas Oil Stadium",                "roof": "dome",      "surface": "turf"},
    "JAX": {"name": "EverBank Stadium",                 "roof": "outdoors",  "surface": "turf"},
    "KC":  {"name": "GEHA Field at Arrowhead Stadium",  "roof": "outdoors",  "surface": "grass"},
    "LAC": {"name": "SoFi Stadium",                     "roof": "dome",      "surface": "turf"},
    "LAR": {"name": "SoFi Stadium",                     "roof": "dome",      "surface": "turf"},
    "LV":  {"name": "Allegiant Stadium",                "roof": "dome",      "surface": "turf"},
    "MIA": {"name": "Hard Rock Stadium",                "roof": "outdoors",  "surface": "grass"},
    "MIN": {"name": "U.S. Bank Stadium",                "roof": "dome",      "surface": "turf"},
    "NE":  {"name": "Gillette Stadium",                 "roof": "outdoors",  "surface": "turf"},
    "NO":  {"name": "Caesars Superdome",                "roof": "dome",      "surface": "turf"},
    "NYG": {"name": "MetLife Stadium",                  "roof": "outdoors",  "surface": "turf"},
    "NYJ": {"name": "MetLife Stadium",                  "roof": "outdoors",  "surface": "turf"},
    "PHI": {"name": "Lincoln Financial Field",          "roof": "outdoors",  "surface": "grass"},
    "PIT": {"name": "Acrisure Stadium",                 "roof": "outdoors",  "surface": "turf"},
    "SEA": {"name": "Lumen Field",                      "roof": "open",      "surface": "turf"},
    "SF":  {"name": "Levi's Stadium",                   "roof": "outdoors",  "surface": "grass"},
    "TB":  {"name": "Raymond James Stadium",            "roof": "outdoors",  "surface": "grass"},
    "TEN": {"name": "Nissan Stadium",                   "roof": "outdoors",  "surface": "turf"},
    "WAS": {"name": "Commanders Field",                 "roof": "outdoors",  "surface": "grass"},
}

def venue_defaults(home_team: str):
    ht = (home_team or "").upper().strip()
    v = STADIUMS.get(ht, {"name": "Unknown stadium", "roof": "outdoors", "surface": "grass"})
    return v["name"], v["roof"], v["surface"]

STADIUMS_DIR = Path(__file__).parent / "team_stadiums"

def stadium_image_path(team: str):
    """
    Return path to the home stadium image for a team, or None if not found.
    Expects files named like 'KC_HOME.png' in team_stadiums/.
    """
    if not team:
        return None
    p = STADIUMS_DIR / f"{team.upper()}_HOME.png"
    return str(p) if p.exists() else None

# sidebar: objective
with st.sidebar:
    st.header("Decision Objective")
    metric = "wpa" if st.radio(
        "Optimize for:", ["Win Probability (WPA)", "Expected Points (EPA)"]
    ).startswith("Win") else "epa"

# game info
st.subheader("Game Info")
g1, g2 = st.columns(2)
with g1:
    season = st.number_input("Season", min_value=2016, max_value=2024, value=2024, step=1)
with g2:
    week = st.number_input("Week", min_value=1, max_value=18, value=12, step=1)

# teams & possession (dropdowns)
st.subheader("Teams & possession")
t1, t2 = st.columns(2)
with t1:
    home_team = st.selectbox("Home team", TEAM_ABBRS, index=TEAM_ABBRS.index("BUF"))
    logo_home = team_logo(home_team)
    if logo_home:
        st.image(logo_home, width=100, caption=TEAM_FULL_NAMES.get(home_team, home_team))
with t2:
    away_team = st.selectbox("Away team", TEAM_ABBRS, index=TEAM_ABBRS.index("KC"))
    logo_away = team_logo(away_team)
    if logo_away:
        st.image(logo_away, width=100, caption=TEAM_FULL_NAMES.get(away_team, away_team))
# sanity check
if home_team == away_team:
    st.warning("Home and Away teams are the same — pick two different teams.")
    st.stop()
# possession dropdown
posteam = st.selectbox("Offense (Possession) Team", [home_team, away_team], index=1)
defteam = away_team if posteam == home_team else home_team

stadium_name, auto_roof, auto_surface = venue_defaults(home_team)  # you already call this later; okay to call here

sp1, sp2, sp3 = st.columns([1, 3, 1])  # center the image
with sp2:
    stadium_pic = stadium_image_path(home_team)
    if stadium_pic:
        st.image(
            stadium_pic,
            caption=f"{stadium_name} — {TEAM_FULL_NAMES.get(home_team, home_team)}",
            use_column_width=True
        )

# venue & weather
stadium_name, auto_roof, auto_surface = venue_defaults(home_team)
with st.expander("Venue & weather", expanded=False):
    # friendly message that mapping was applied (with stadium name)
    st.info(
        f"{TEAM_FULL_NAMES.get(home_team, home_team)} home stadium: **{stadium_name}**. "
        f"Defaults applied → roof: **{auto_roof}**, surface: **{auto_surface}**. "
        "You can override below."
    )

    roof = st.selectbox(
        "Roof", ["outdoors", "dome", "open"],
        index=["outdoors","dome","open"].index(auto_roof)
    )
    surface = st.selectbox(
        "Surface", ["grass","turf"],
        index=["grass","turf"].index(auto_surface)
    )

    wcol = st.columns(2)
    with wcol[0]:
        temp_default = 70 if roof == "dome" else 60
        temp = st.number_input("Temp (°F)", value=temp_default, min_value=-10, max_value=120)
    with wcol[1]:
        wind_default = 0 if roof == "dome" else 5
        wind = st.number_input("Wind (mph)", value=wind_default, min_value=0, max_value=50)

    st.caption("Note: Temperature and wind affect predictions only if the model was trained with these features.")

# situation
st.subheader("4th Down Situation")
c0, c1, c2, c3 = st.columns([1,1.1,1.1,1.1])
with c0:
    qtr = st.selectbox("Quarter", [1,2,3,4], index=3)
with c1:
    mins_left = st.number_input("Minutes left in quarter", 0, 15, 2)
with c2:
    secs_left = st.number_input("Seconds left in quarter", 0, 59, 0)
with c3:
    ydstogo = st.number_input("Yards to go", 1, 50, 1)

# Score
s1, s2 = st.columns(2)
with s1:
    score_off = st.number_input("Offense score", 0, 80, 0)
with s2:
    score_def = st.number_input("Defense score", 0, 80, 0)

# yardline helper
c4, c5 = st.columns([1,2])
with c4:
    side = st.selectbox("Ball on", ["OWN side","OPP side"], index=1)
    yard_line = st.number_input("Yard line (1–49)", 1, 49, 48)
with c5:
    st.image("field_diagram.png", caption="Example: 48 yd line, enemy territory near midfield → 'OPP side', 48. 10 yd line, deep in own territory → 'OWN side', 10.", use_column_width=True)

# convert to yardline_100
yardline_100 = (50 + yard_line) if side == "OWN side" else (50 - yard_line)
yardline_100 = int(max(1, min(99, yardline_100)))

# timeouts
st.subheader("Timeouts Remaining")
to1, to2 = st.columns(2)
with to1:
    home_timeouts_remaining = st.number_input("Home timeouts", 0, 3, 3)
with to2:
    away_timeouts_remaining = st.number_input("Away timeouts", 0, 3, 3)

# derive posteam/defteam timeouts
if posteam == home_team:
    posteam_timeouts_remaining = home_timeouts_remaining
    defteam_timeouts_remaining = away_timeouts_remaining
else:
    posteam_timeouts_remaining = away_timeouts_remaining
    defteam_timeouts_remaining = home_timeouts_remaining

# derived fields
quarter_len = 15 * 60
game_seconds_remaining = (4 - qtr) * quarter_len + int(mins_left) * 60 + int(secs_left)
score_diff = int(score_off) - int(score_def)

# goal-to-go
goal_to_go = int((yardline_100 <= 10) and (ydstogo >= yardline_100))  # close proxy to pbp's flag

# auto-fill offense team metrics from CSV
auto = get_team_feats(int(season), int(week), posteam)
with st.expander("Auto-filled team metrics (offense)", expanded=False):
    metrics = [
        ["Offense EPA (last 4w)", f"{auto['off_epa_4w']:.3f}"],
        ["Defense EPA (last 4w)", f"{auto['def_epa_4w']:.3f}"],
        ["FG % (short)", f"{auto['fg_pct_short']*100:.0f}%"],
        ["FG % (mid)", f"{auto['fg_pct_mid']*100:.0f}%"],
        ["FG % (long)", f"{auto['fg_pct_long']*100:.0f}%"],
        ["Punt net yards (4w avg)", f"{auto['punt_net_4w']:.1f}"],
    ]
    metrics_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
    st.table(metrics_df)

    st.caption(f"Source: {posteam}, season {season}, week {week}")

# hidden neutral defaults for features you chose not to collect live
plays_in_drive_so_far = 3
elapsed_game = 4 * quarter_len - game_seconds_remaining
def_time_on_field_cum = int(0.5 * elapsed_game)
def_time_on_field_share = 0.5

# build model context (only include keys the model knows)
ctx = {
    "yardline_100": yardline_100,
    "ydstogo": int(ydstogo),
    "score_differential": score_diff,
    "qtr": int(qtr),
    "game_seconds_remaining": int(game_seconds_remaining),

    "off_epa_4w": auto["off_epa_4w"],
    "def_epa_4w": auto["def_epa_4w"],
    "fg_pct_short": auto["fg_pct_short"],
    "fg_pct_mid": auto["fg_pct_mid"],
    "fg_pct_long": auto["fg_pct_long"],
    "punt_net_4w": auto["punt_net_4w"],

    "plays_in_drive_so_far": plays_in_drive_so_far,
    "def_time_on_field_cum": def_time_on_field_cum,
    "def_time_on_field_share": def_time_on_field_share,

    "posteam": posteam, "defteam": defteam,
    "home_team": home_team, "away_team": away_team,
    "roof": roof, "surface": surface,
}

# conditionally include extras IF the model was trained with them
maybe = {
    "home_timeouts_remaining": home_timeouts_remaining,
    "away_timeouts_remaining": away_timeouts_remaining,
    "posteam_timeouts_remaining": posteam_timeouts_remaining,
    "defteam_timeouts_remaining": defteam_timeouts_remaining,
    "temp": temp,
    "wind": wind,
    "goal_to_go": goal_to_go,
}
ctx.update({k: v for k, v in maybe.items() if k in FEATURE_COLS})

kick_dist = int(17 + yardline_100)
st.caption(f"Estimated FG distance from here: ~{kick_dist} yards.")


# recommend
if st.button("Recommend decision"):
    epa_scores, wpa_scores, rec, _ = score_context(ctx, metric=metric)

    # pick metric dict
    scores = wpa_scores if metric == "wpa" else epa_scores

    # filter out infeasible actions (masked in inference as huge negative)
    SENTINEL = -1e8  # must be > the mask used in inference (-1e9 there)
    feasible = {a: v for a, v in scores.items() if v is not None and v > SENTINEL}
    infeasible = [a for a in scores if a not in feasible]

    # safety: if rec is infeasible for any reason, pick best feasible
    if rec not in feasible:
        rec = max(feasible, key=feasible.get)

    # rank feasible only
    ranked = sorted(feasible.items(), key=lambda kv: kv[1], reverse=True)
    best_action, best_val = ranked[0]

    # build html for box 
    title_html = f"<h3 style='margin:0;'>Recommendation: <b>{best_action.upper()}</b> <i>(optimized for {metric.upper()})</i></h3>"

    if len(ranked) >= 2:
        second_action, second_val = ranked[1]
        if metric == "wpa":
            lead_html = (
                f"<p><b>{best_action.upper()}</b> improves win probability by "
                f"<b>{(best_val - second_val):.1%}</b> vs <b>{second_action.upper()}</b>.</p>"
            )
        else:
            lead_html = (
                f"<p><b>{best_action.upper()}</b> improves expected points by "
                f"<b>{(best_val - second_val):.2f} EPA</b> vs <b>{second_action.upper()}</b>.</p>"
            )
    else:
        lead_html = "<p>Only one feasible option in this situation.</p>"

    # friendly bullets vs *each* alternative (feasible only)
    bullets = []
    for a, v in ranked[1:]:
        if metric == "wpa":
            bullets.append(f"{best_action.upper()} vs {a.upper()}: <b>{(best_val - v):.1%} WPA</b>")
        else:
            bullets.append(f"{best_action.upper()} vs {a.upper()}: <b>{(best_val - v):.2f} EPA</b>")

    bullets_html = ""
    if bullets:
        bullets_html = (
            "<p style='margin:0 0 4px 0; color:#555;'>Relative gains:</p>"
            "<ul style='margin-top:4px;'>" +
            "".join(f"<li>{b}</li>" for b in bullets) +
            "</ul>"
        )

    # put the message right here
    msg_html = ""
    if infeasible:
        msg = "⚠️ Excluded as infeasible: " + ", ".join(x.upper() for x in infeasible)
        if "fg" in infeasible:
            msg += " (field goal out of realistic range)"
        if "punt" in infeasible:
            msg += " (punting not considered in this situation)"
        msg_html = f"<p style='color:#777;'>{msg}</p>"

    # Final box style
    box_html = f"""
    <div style="
      border:2px solid #4CAF50;
      border-radius:12px;
      padding:16px 18px;
      margin:12px 0;
      background:#f9f9f9;">
      {title_html}
      {lead_html}
      {bullets_html}
      {msg_html}
    </div>
    """

    st.markdown(box_html, unsafe_allow_html=True)