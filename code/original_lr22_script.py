# Linear Regression 2.2 - Dataset builder and validator (Krishna)
# Input : post_match_details.csv (104 matches)  [+ player_stats.csv for a cross-check]
# Output: lr22_dataset.csv (208 rows = 2 team records per match)

import pandas as pd
import numpy as np

IN = "/Users/krishnabhatta/Desktop/lr22/"
OUT = "/Users/krishnabhatta/Desktop/lr22/" 

d = pd.read_csv(IN + "post_match_details.csv")

d["event_dt"] = pd.to_datetime(d["event_date"], utc=True)
for c in ["extra_time_score_home", "extra_time_score_away"]:
    d[c] = d[c].fillna(0)            

d["home_goals"] = d["home_score"] + d["extra_time_score_home"]
d["away_goals"] = d["away_score"] + d["extra_time_score_away"]

d["is_knockout"] = d["round_name"].notna().astype(int)
d["stage"] = np.where(d["round_name"].isna(), d["group_name"], d["round_name"])
HOSTS = {"USA", "Mexico", "Canada"}


home = pd.DataFrame({
    "match_id": d["event_id"], "match_datetime_utc": d["event_dt"], "stage": d["stage"],
    "is_knockout": d["is_knockout"], "team": d["home_team"], "opponent": d["away_team"],
    "goals_scored": d["home_goals"], "goals_conceded": d["away_goals"], "listed_as": "home"})
away = pd.DataFrame({
    "match_id": d["event_id"], "match_datetime_utc": d["event_dt"], "stage": d["stage"],
    "is_knockout": d["is_knockout"], "team": d["away_team"], "opponent": d["home_team"],
    "goals_scored": d["away_goals"], "goals_conceded": d["home_goals"], "listed_as": "away"})
long = pd.concat([home, away], ignore_index=True)
long = long.sort_values(["match_datetime_utc", "match_id", "listed_as"]).reset_index(drop=True)


long["points"] = np.select([long.goals_scored > long.goals_conceded,
                            long.goals_scored == long.goals_conceded], [3, 1], 0)
long = long.sort_values(["team", "match_datetime_utc", "match_id"])
g = long.groupby("team")
long["team_matches_played_before"] = g.cumcount()
long["team_goals_scored_before"] = g["goals_scored"].cumsum() - long["goals_scored"]
long["team_goals_conceded_before"] = g["goals_conceded"].cumsum() - long["goals_conceded"]
long["team_points_before"] = g["points"].cumsum() - long["points"]
long = long.sort_values(["match_datetime_utc", "match_id", "listed_as"]).reset_index(drop=True)


opp = long[["match_id", "team", "team_goals_scored_before", "team_goals_conceded_before"]].rename(
    columns={"team": "opponent", "team_goals_scored_before": "opp_goals_scored_before",
             "team_goals_conceded_before": "opp_goals_conceded_before"})
long = long.merge(opp, on=["match_id", "opponent"], how="left")

long["is_host"] = long["team"].isin(HOSTS).astype(int)
long = long.sort_values(["match_datetime_utc", "match_id", "listed_as"]).reset_index(drop=True)
long.insert(0, "row_id", range(1, len(long) + 1))

ID_COLS = ["row_id", "match_id", "match_datetime_utc", "stage", "team", "opponent"]
TARGET = ["goals_scored"]
X = ["is_host", "is_knockout", "team_matches_played_before", "team_goals_scored_before",
     "team_goals_conceded_before", "opp_goals_scored_before", "opp_goals_conceded_before",
     "team_points_before"]
final = long[ID_COLS + TARGET + X].copy()
final["goals_scored"] = final["goals_scored"].astype(int)
final["match_datetime_utc"] = final["match_datetime_utc"].dt.strftime("%Y-%m-%d %H:%M")
final.to_csv(OUT + "lr22_dataset.csv", index=False)


print("Shape:", final.shape, "(expected (208, 15))")
print("Explanatory variables:", len(X))
print("Missing values:", int(final.isnull().sum().sum()))
print("Duplicate row_id:", final.row_id.duplicated().sum(),
      "| duplicate (match_id, team):", final.duplicated(["match_id", "team"]).sum())
print("Rows per match (all must be 2):", final.groupby("match_id").size().unique())
print("Matches:", final.match_id.nunique(), "| Teams:", final.team.nunique())
print("Goals integer & >=0:", bool((final.goals_scored >= 0).all()))
chk = final.merge(final, left_on=["match_id", "team"], right_on=["match_id", "opponent"], suffixes=("", "_o"))
print("Opponent's 'before' values mirror correctly:",
      bool((chk.opp_goals_scored_before == chk.team_goals_scored_before_o).all()))
print("First-match rows have all-zero history:",
      bool((final[final.team_matches_played_before == 0][X[2:5] + [X[7]]] == 0).all().all()))
print("Total goals in dataset:", final.goals_scored.sum())
print("\nGoals scored distribution:\n", final.goals_scored.value_counts().sort_index().to_string())
print("\nDescriptive stats:\n", final[TARGET + X].describe().round(3).T.to_string())
print("\nCorrelation matrix of explanatory variables:\n", final[X].corr().round(2).to_string())


p = pd.read_csv(IN + "player_stats.csv")
print("\nCross-check: dataset goals =", int(final.goals_scored.sum()),
      "| player file goals + own goals =", int(p.goals.sum() + p.own_goals.sum()))
