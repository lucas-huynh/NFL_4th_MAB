from inference import score_context

ctx = {
  'yardline_100': 52, 'ydstogo': 1, 'score_differential': -3,
  'qtr': 4, 'game_seconds_remaining': 120,  # 2:00 left

  # team form / ST
  'off_epa_4w': 0.1, 'def_epa_4w': -0.05,
  'fg_pct_short': 0.90, 'fg_pct_mid': 0.80, 'fg_pct_long': 0.60,
  'punt_net_4w': 42,

  # drive/fatigue
  'plays_in_drive_so_far': 5,
  'def_time_on_field_cum': 900, 'def_time_on_field_share': 0.56,

  # timeouts (required by fitted preprocessor)
  'home_timeouts_remaining': 3,
  'away_timeouts_remaining': 2,
  'posteam_timeouts_remaining': 2,  # if offense is away here
  'defteam_timeouts_remaining': 3,

  # weather (required by fitted preprocessor)
  'temp': 55,         # °F
  'wind': 10,         # mph

  # situational (required by fitted preprocessor)
  'goal_to_go': 0,    # 1 if yards-to-go >= distance to goal line

  # teams / venue
  'posteam': 'KC', 'defteam': 'BUF', 'home_team': 'BUF', 'away_team': 'KC',
  'posteam_type': 'away', 'roof': 'outdoors', 'surface': 'grass',
}

epa_scores, wpa_scores, rec, _ = score_context(ctx, metric="wpa")
print("Recommendation:", rec)
print("WPA scores:", wpa_scores)
print("EPA scores:", epa_scores)
