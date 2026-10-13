"""
Step 1 - Building the two datasets.

    LR 2.1 : one row per match (104 rows),          response = goal difference (home - away)
    LR 2.2 : one row per team per match (208 rows), response = goals scored by the team

"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

HOST_COUNTRY = {"USA": "United States", "Canada": "Canada", "Mexico": "Mexico"}


def km_between(lat1, lon1, lat2, lon2):
    """Distance in km between two points on the Earth (haversine formula)."""
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------------------------------
# 1. Load the two files, join them, and put the matches in time order
# ------------------------------------------------------------------------------------------
details = pd.read_csv(RAW / "post_match_details.csv")
stats = pd.read_csv(RAW / "post_match_stats.csv")
m = details.merge(stats, on="event_id")
m["kickoff"] = pd.to_datetime(m["event_date"], utc=True)
m = m.sort_values(["kickoff", "event_id"]).reset_index(drop=True)
m["match_id"] = m.index + 1                                            # 1 to 104

m["stage"] = m["round_name"].fillna(m["group_name"])                   # e.g. "Group A" or "Round of 32"
m["knockout_stage"] = m["round_name"].notna().astype(int)              # 1 for knockout matches
m["temperature_c"] = m["weather"].apply(lambda w: json.loads(w)["temperature_c"])
card_columns = ["home_yellow_cards", "home_red_cards", "away_yellow_cards", "away_red_cards"]
m[card_columns] = m[card_columns].fillna(0)                            # blank = no card

# ------------------------------------------------------------------------------------------
# 2. One row per team per match (104 matches x 2 teams = 208 rows)
# ------------------------------------------------------------------------------------------
def team_rows(side, other):
    """The numbers of one team in every match, from that team's point of view."""
    return pd.DataFrame({
        "match_id": m["match_id"], "kickoff": m["kickoff"], "venue_id": m["venue_id"],
        "team": m[side + "_team"], "is_home": int(side == "home"),
        "goals": m[side + "_score"],
        "xg_for": m[side + "_expected_goals"],
        "xg_against": m[other + "_expected_goals"],
        "xg_net": m[side + "_expected_goals"] - m[other + "_expected_goals"],
        "finishing": m[side + "_score"] - m[side + "_expected_goals"],          # goals minus xG
        "keeper_prevented": m[side + "_goals_prevented"],
        "final_third_entries": m[side + "_final_third_entries"],
        "box_touches": m[side + "_touches_in_penalty_area"],
        "sot_faced": m[other + "_shots_on_target"],                              # shots on target against this team
        "fair_play_lost": m[side + "_yellow_cards"] + 4 * m[side + "_red_cards"],
    })


rows = pd.concat([team_rows("home", "away"), team_rows("away", "home")])
rows = rows.sort_values(["team", "kickoff"]).reset_index(drop=True)
venues = pd.read_csv(RAW / "venues.csv")                              # made by 00_get_venues.py (city, country, lat, lon)
rows = rows.merge(venues[["venue_id", "country", "latitude", "longitude"]], on="venue_id", how="left", validate="m:1")
rows = rows.rename(columns={"latitude": "lat", "longitude": "lon"})
rows["host"] = (rows["team"].map(HOST_COUNTRY) == rows["country"]).astype(int)   # host nation playing in its own country

# ------------------------------------------------------------------------------------------
# 3. "Before the match" variables: average over the team's EARLIER matches only
# ------------------------------------------------------------------------------------------
by_team = rows.groupby("team")
matches_before = by_team.cumcount()                                    # 0 for a team's first match, then 1, 2, ...
for col in ["xg_for", "xg_against", "xg_net", "finishing", "keeper_prevented",
            "final_third_entries", "box_touches", "sot_faced", "fair_play_lost"]:
    total_before = by_team[col].cumsum() - rows[col]                   # total so far, minus the current match
    rows["avg_" + col] = total_before / matches_before.replace(0, np.nan)   # average of the earlier matches

rows["rest_days"] = by_team["kickoff"].diff().dt.total_seconds() / 86400          # days since last match
rows["travel_km"] = km_between(by_team["lat"].shift(1), by_team["lon"].shift(1), rows["lat"], rows["lon"])
# (a team's first match has no earlier match, so these values are empty - handled below)

# ------------------------------------------------------------------------------------------
# 4. Put each team's row next to its OPPONENT's row (columns ending in _opp)
# ------------------------------------------------------------------------------------------
pairs = rows.merge(rows, on="match_id", suffixes=("", "_opp"))
pairs = pairs[pairs["team"] != pairs["team_opp"]]                       # keep team vs opponent, not team vs itself
match_info = m[["match_id", "event_id", "stage", "knockout_stage", "temperature_c"]]
pairs = pairs.merge(match_info, on="match_id").sort_values(["match_id", "team"]).reset_index(drop=True)
pairs["date"] = pairs["kickoff"].dt.strftime("%Y-%m-%d")

# ------------------------------------------------------------------------------------------
# 5. LR 2.1 - one row per match, variables = HOME minus AWAY
# ------------------------------------------------------------------------------------------
home = pairs[pairs["is_home"] == 1].reset_index(drop=True)              # 104 rows: home team vs away team
lr21 = pd.DataFrame({
    "match_id": home["match_id"], "event_id": home["event_id"], "date": home["date"], "stage": home["stage"],
    "home_team": home["team"], "away_team": home["team_opp"],
    "goal_difference": home["goals"] - home["goals_opp"],                              # RESPONSE
    "xg_net_diff": home["avg_xg_net"] - home["avg_xg_net_opp"],
    "finishing_diff": home["avg_finishing"] - home["avg_finishing_opp"],
    "keeper_goals_prevented_diff": home["avg_keeper_prevented"] - home["avg_keeper_prevented_opp"],
    "final_third_entries_diff": home["avg_final_third_entries"] - home["avg_final_third_entries_opp"],
    "fair_play_points_lost_diff": home["avg_fair_play_lost"] - home["avg_fair_play_lost_opp"],
    "travel_km_diff": home["travel_km"] - home["travel_km_opp"],
    "rest_days_diff": home["rest_days"] - home["rest_days_opp"],
    "host_advantage": home["host"] - home["host_opp"],
})
x21 = lr21.columns[7:]                                                   # the 8 explanatory variables
lr21[x21] = lr21[x21].fillna(0)                      # no earlier match -> treat as an average team (difference 0)
lr21[x21[:7]] = lr21[x21[:7]].round(3)
lr21.to_csv(OUT / "lr21_dataset.csv", index=False)

# ------------------------------------------------------------------------------------------
# 6. LR 2.2 - one row per team per match, using the team's own numbers and the opponent's
# ------------------------------------------------------------------------------------------
lr22 = pd.DataFrame({
    "match_id": pairs["match_id"], "event_id": pairs["event_id"], "date": pairs["date"], "stage": pairs["stage"],
    "team": pairs["team"], "opponent": pairs["team_opp"],
    "goals_scored": pairs["goals"],                                                    # RESPONSE
    "team_xg_for_avg": pairs["avg_xg_for"].fillna(rows["xg_for"].mean()),
    "opp_xg_against_avg": pairs["avg_xg_against_opp"].fillna(rows["xg_against"].mean()),
    "opp_keeper_goals_prevented_avg": pairs["avg_keeper_prevented_opp"].fillna(0),
    "team_box_touches_avg": pairs["avg_box_touches"].fillna(rows["box_touches"].mean()),
    "opp_shots_on_target_faced_avg": pairs["avg_sot_faced_opp"].fillna(rows["sot_faced"].mean()),
    "temperature_c": pairs["temperature_c"],
    "knockout_stage": pairs["knockout_stage"],
    "team_is_host": pairs["host"],
})
x22 = lr22.columns[7:]
lr22[x22[:5]] = lr22[x22[:5]].round(3)               # first-round teams get the tournament average instead
lr22.to_csv(OUT / "lr22_dataset.csv", index=False)

# ------------------------------------------------------------------------------------------
# 7. Responses including extra-time goals (only used for a robustness check in step 2)
# ------------------------------------------------------------------------------------------
extra = pd.DataFrame({
    "match_id": m["match_id"],
    "home_goals_incl_et": m["home_score"] + m["extra_time_score_home"].fillna(0),
    "away_goals_incl_et": m["away_score"] + m["extra_time_score_away"].fillna(0),
})
extra.to_csv(OUT / "extra_time_targets.csv", index=False)

# ------------------------------------------------------------------------------------------
# Quick checks
# ------------------------------------------------------------------------------------------
assert lr21.shape[0] == 104 and lr22.shape[0] == 208 and len(x21) == 8 and len(x22) == 8
assert lr21[x21].notna().all().all() and lr22[x22].notna().all().all()
print("LR 2.1 dataset:", lr21.shape, "  LR 2.2 dataset:", lr22.shape)
