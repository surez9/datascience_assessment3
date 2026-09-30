
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # save figures without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# setup-
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = DATA_DIR
DATA_FILE = os.path.join(DATA_DIR, "lr21_dataset.csv")

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


#loading the dataset


d = pd.read_csv(DATA_FILE)
X_cols = ["rank_gap", "form_gf_diff", "form_ga_diff", "form_pts_diff",
          "rest_days_diff", "host_adv", "strong_confed_diff", "debutant_diff"]
TARGET = "goal_diff"
print(f"Loaded {DATA_FILE}: {d.shape[0]} rows x {d.shape[1]} columns")
print(d.head(10).to_string())


section("2. VALIDATION")

teams = set(d.home_team) | set(d.away_team)
n_missing = int(d[X_cols + [TARGET]].isna().sum().sum())
numeric_ok = all(pd.api.types.is_numeric_dtype(d[c]) for c in X_cols + [TARGET])
coded_ok = all(d[c].isin([-1, 0, 1]).all() for c in ["host_adv", "strong_confed_diff", "debutant_diff"])
# Leakage check: when BOTH teams play their first match, no earlier data exists,
# so form and rest variables must equal 0. File order is chronological (kick-off time).
seen, first_rows = set(), []
for i, r in d.iterrows():
    if r.home_team not in seen and r.away_team not in seen:
        first_rows.append(i)
    seen.update([r.home_team, r.away_team])
form_ok = (d.loc[first_rows, ["form_gf_diff", "form_ga_diff", "form_pts_diff", "rest_days_diff"]] == 0).all().all()
checks = pd.DataFrame({
    "Check": ["Row count = 104", "Explanatory variables = 8", "Unique match IDs",
              "No duplicate fixtures", "Missing values", "All model columns numeric",
              "Indicator variables coded -1/0/1", f"First-match form and rest = 0 ({len(first_rows)} rows, no leakage)",
              "Teams in dataset", "Goal difference range plausible (-10 to 10)"],
    "Result": [len(d), len(X_cols), d.match_id.is_unique,
               not d.duplicated(["date", "home_team", "away_team"]).any(), n_missing, numeric_ok,
               coded_ok, form_ok, len(teams), f"{d[TARGET].min()} to {d[TARGET].max()}"],
    "Pass": [len(d) == 104, len(X_cols) == 8, d.match_id.is_unique,
             not d.duplicated(["date", "home_team", "away_team"]).any(), n_missing == 0, numeric_ok,
             coded_ok, form_ok, len(teams) == 48, d[TARGET].between(-10, 10).all()],
})
save_table(checks, "T1_validation_checks", index=False)
if not checks.Pass.all():
    print("\nWARNING: at least one validation check failed. Review the table above.")

stage_counts = d.stage.where(~d.stage.str.startswith("Group"), "Group stage").value_counts()
print("\nMatches by stage:\n" + stage_counts.to_string())


# 3. EXPLORATORY DATA ANALYSIS
section("3. EXPLORATORY DATA ANALYSIS")

desc = d[X_cols + ["goal_diff"]].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
save_table(desc, "T2_descriptive_statistics")

print(f"\nGoal difference: mean {d.goal_diff.mean():.2f}, SD {d.goal_diff.std():.2f}, "
      f"skew {d.goal_diff.skew():.2f}, draws {(d.goal_diff == 0).mean():.0%}")

# F1: target distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
vc = d.goal_diff.value_counts().sort_index()
ax[0].bar(vc.index, vc.values, color="#2E86AB")
ax[0].set(title="Distribution of goal difference (home - away)", xlabel="Goal difference", ylabel="Matches")
for x, yv in zip(vc.index, vc.values):
    ax[0].text(x, yv + 0.4, yv, ha="center", fontsize=9)
outcome = pd.Series(np.select([d.goal_diff > 0, d.goal_diff < 0],
                              ["Home-listed win", "Away-listed win"], "Draw")).value_counts()
ax[1].bar(outcome.index, outcome.values, color=["#2E86AB", "#A23B72", "#F18F01"])
ax[1].set(title="Match outcomes after 90 minutes", ylabel="Matches")
for i, v in enumerate(outcome.values):
    ax[1].text(i, v + 0.5, v, ha="center")
save_fig("F1_target_distribution")

# F2: correlation heatmap
plt.figure(figsize=(9, 7))
corr = d[X_cols + ["goal_diff"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": .8})
plt.title("Correlation matrix: explanatory variables and goal difference")
save_fig("F2_correlation_heatmap")
target_corr = (corr["goal_diff"].drop("goal_diff")
               .sort_values(key=abs, ascending=False).round(3).to_frame("r with goal_diff"))
save_table(target_corr, "T3_target_correlations")

# F3: scatter of each variable vs target
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
rng = np.random.default_rng(1)
for a, c in zip(axes.flat, X_cols):
    jitter = rng.normal(0, .12, len(d))
    a.scatter(d[c], d.goal_diff + jitter, alpha=.55, s=22, color="#2E86AB")
    b = np.polyfit(d[c], d.goal_diff, 1)
    xs = np.linspace(d[c].min(), d[c].max(), 50)
    a.plot(xs, np.polyval(b, xs), color="#C73E1D", lw=2)
    a.set(title=f"{c}  (r = {d[c].corr(d.goal_diff):.2f})", xlabel=c, ylabel="Goal difference")
save_fig("F3_scatter_each_variable")

# F4: rank gap vs goal difference
plt.figure(figsize=(9, 5.5))
sns.regplot(data=d, x="rank_gap", y="goal_diff", scatter_kws={"alpha": .6, "s": 30},
            line_kws={"color": "#C73E1D"})
labels = {("Germany", "Curaçao"): (-40, -38), ("Canada", "Qatar"): (-110, -10),
          ("Senegal", "Iraq"): (-120, -6), ("Portugal", "Uzbekistan"): (25, -48),
          ("New Zealand", "Belgium"): (10, 10), ("Brazil", "Haiti"): (-95, -35)}
for (home, away), offset in labels.items():
    row = d[(d.home_team == home) & (d.away_team == away)]
    if row.empty:
        continue
    row = row.iloc[0]
    plt.annotate(f"{home} v {away} ({row.goal_diff:+d})", (row.rank_gap, row.goal_diff),
                 fontsize=8, xytext=offset, textcoords="offset points",
                 arrowprops=dict(arrowstyle="-", color="grey", lw=.7))
plt.title("FIFA rank gap vs goal difference")
plt.xlabel("Away rank - home rank (positive = home team higher ranked)")
plt.ylabel("Goal difference")
save_fig("F4_rank_gap_vs_goal_diff")

# Multicollinearity
X_const = sm.add_constant(d[X_cols])
vif = pd.DataFrame({"Variable": X_cols,
                    "VIF": [variance_inflation_factor(X_const.values, i + 1) for i in range(len(X_cols))]}).round(2)
save_table(vif, "T4_vif", index=False)


# 4. BUILD THE MODEL 
section("4. MODEL: OLS ON ALL 104 MATCHES")

X, y = d[X_cols], d[TARGET]
ols = sm.OLS(y, sm.add_constant(X)).fit()
print(ols.summary())

robust = ols.get_robustcov_results(cov_type="HC3")        # heteroscedasticity-robust SEs
coef = pd.DataFrame({
    "Coefficient": ols.params,
    "Std error": ols.bse,
    "t": ols.tvalues,
    "p-value": ols.pvalues,
    "CI 2.5%": ols.conf_int()[0],
    "CI 97.5%": ols.conf_int()[1],
    "Robust (HC3) p-value": robust.pvalues,
})
coef["Std. coefficient"] = ols.params * pd.concat([pd.Series({"const": np.nan}), X.std()]) / y.std()
pvals = coef[["p-value", "Robust (HC3) p-value"]].copy()
coef = coef.round(4)
coef[["p-value", "Robust (HC3) p-value"]] = pvals.round(7)
save_table(coef, "T5_coefficients")