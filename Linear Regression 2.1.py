import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = DATA_DIR
MATCH_FILE = os.path.join(DATA_DIR, "Dataset/post_match_details.csv")
PLAYER_FILE = os.path.join(DATA_DIR, "Dataset/player_stats.csv")

FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{name}.png")


def save_table(df, name, index=True):
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=index)
    print(f"\n--- {name} ---")
    print(df.to_string())


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)



section("1. BUILD DATASET")

m = pd.read_csv(MATCH_FILE)
m["date"] = pd.to_datetime(m["event_date"])
m = m.sort_values("date").reset_index(drop=True)

# FIFA ranking published 11 June 2026 (tournament opening day): pre-match info
fifa_rank = {
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4, "Portugal": 5, "Brazil": 6,
    "Morocco": 7, "Netherlands": 8, "Belgium": 9, "Germany": 10, "Croatia": 11,
    "Colombia": 13, "Mexico": 14, "Senegal": 15, "Uruguay": 16, "USA": 17, "Japan": 18,
    "Switzerland": 19, "Iran": 20, "Türkiye": 22, "Ecuador": 23, "Austria": 24,
    "South Korea": 25, "Australia": 27, "Algeria": 28, "Egypt": 29, "Canada": 30,
    "Norway": 31, "Côte d'Ivoire": 33, "Panama": 34, "Sweden": 38, "Czechia": 40,
    "Paraguay": 41, "Scotland": 42, "Tunisia": 45, "DR Congo": 46, "Uzbekistan": 50,
    "Qatar": 56, "Iraq": 57, "South Africa": 60, "Saudi Arabia": 61, "Jordan": 63,
    "Bosnia & Herzegovina": 64, "Cabo Verde": 67, "Ghana": 73, "Curaçao": 82,
    "Haiti": 83, "New Zealand": 85,
}
conmebol = {"Argentina", "Brazil", "Colombia", "Uruguay", "Ecuador", "Paraguay"}
uefa = {"Spain", "France", "England", "Portugal", "Netherlands", "Belgium", "Germany",
        "Croatia", "Switzerland", "Türkiye", "Austria", "Norway", "Sweden", "Czechia",
        "Scotland", "Bosnia & Herzegovina"}
hosts = {"USA", "Mexico", "Canada"}
debutants = {"Cabo Verde", "Curaçao", "Jordan", "Uzbekistan"}   # first World Cup in 2026

assert set(fifa_rank) == set(m.home_team) | set(m.away_team), "Team names do not match ranking table"

# Final score incl. extra time (penalties excluded). Used ONLY to build form from earlier matches.
m["final_h"] = m.home_score + m.extra_time_score_home.fillna(0)
m["final_a"] = m.away_score + m.extra_time_score_away.fillna(0)

# Rolling pre-match tournament form: each row uses only matches finished BEFORE it
history = {team: [] for team in fifa_rank}      # list of (date, goals_for, goals_against, points)
form_rows = []
for _, r in m.iterrows():
    feats = {}
    for side, team in (("home", r.home_team), ("away", r.away_team)):
        past = history[team]
        n = len(past)
        feats[f"{side}_gf_pm"] = np.mean([p[1] for p in past]) if n else 0.0
        feats[f"{side}_ga_pm"] = np.mean([p[2] for p in past]) if n else 0.0
        feats[f"{side}_pts_pm"] = np.mean([p[3] for p in past]) if n else 0.0
        feats[f"{side}_rest"] = (r.date - past[-1][0]).total_seconds() / 86400 if n else np.nan
    form_rows.append(feats)

    gh, ga = r.final_h, r.final_a
    ph, pa = (3, 0) if gh > ga else (0, 3) if gh < ga else (1, 1)
    history[r.home_team].append((r.date, gh, ga, ph))
    history[r.away_team].append((r.date, ga, gh, pa))
f = pd.DataFrame(form_rows)


def strong_confed(team):
    return 1 if team in uefa or team in conmebol else 0


d = pd.DataFrame({
    "match_id": m.event_id,
    "date": m.date.dt.date,
    "stage": np.where(m.group_name.notna(), m.group_name, m.round_name),
    "home_team": m.home_team,
    "away_team": m.away_team,
    # ---- 8 explanatory variables (home minus away) ----
    "rank_gap": m.away_team.map(fifa_rank) - m.home_team.map(fifa_rank),
    "form_gf_diff": (f.home_gf_pm - f.away_gf_pm).round(3),
    "form_ga_diff": (f.home_ga_pm - f.away_ga_pm).round(3),
    "form_pts_diff": (f.home_pts_pm - f.away_pts_pm).round(3),
    "rest_days_diff": (f.home_rest - f.away_rest).fillna(0).round(2),
    "host_adv": m.home_team.isin(hosts).astype(int) - m.away_team.isin(hosts).astype(int),
    "strong_confed_diff": m.home_team.map(strong_confed) - m.away_team.map(strong_confed),
    "debutant_diff": m.home_team.isin(debutants).astype(int) - m.away_team.isin(debutants).astype(int),
    # ---- target ----
    "goal_diff": m.home_score - m.away_score,
})
d.to_csv(os.path.join(OUT_DIR, "lr21_dataset.csv"), index=False)
print("Saved lr21_dataset.csv")
print(d.head(10).to_string())

X_cols = ["rank_gap", "form_gf_diff", "form_ga_diff", "form_pts_diff",
          "rest_days_diff", "host_adv", "strong_confed_diff", "debutant_diff"]


# Leakage evidence
print("\nLeakage check 1: head-to-head columns include the match itself (excluded):")
print(m.loc[m.head_to_head_total_matches.notna(),
            ["home_team", "away_team", "home_score", "away_score",
             "head_to_head_total_matches", "head_to_head_home_goals",
             "head_to_head_away_goals"]].head(5).to_string())
if os.path.exists(PLAYER_FILE):
    ps = pd.read_csv(PLAYER_FILE)
    print("\nLeakage check 2: player_stats last_verified =", list(ps.last_verified.unique()),
          "-> end-of-tournament totals (excluded)")
nonzero = int((d.loc[:23, ["form_gf_diff", "form_ga_diff", "form_pts_diff",
                           "rest_days_diff"]].abs().sum(axis=1) > 0).sum())
print(f"\nLeakage check 3: matchday-1 rows with non-zero form values = {nonzero} (expected 0)")