"""
Linear Regression 2.1 (Improved): Predicting Goal Difference in FIFA World Cup 2026 Matches

"""

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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Setup
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MATCH_FILE = os.path.join(DATA_DIR, "post_match_details.csv")
V1_FILE = os.path.join(DATA_DIR, "lr21_dataset.csv")
FIG_DIR = os.path.join(DATA_DIR, "figures")
TAB_DIR = os.path.join(DATA_DIR, "tables")
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


def make_ridge():
    # Ridge regression: scale features, then pick the penalty (alpha) by internal CV
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 50)))


# 1. Build the improved dataset (all features pre-match)
section("1. BUILD IMPROVED DATASET")

m = pd.read_csv(MATCH_FILE)
m["date"] = pd.to_datetime(m["event_date"])
m = m.sort_values("date").reset_index(drop=True)
v1 = pd.read_csv(V1_FILE)
assert len(m) == len(v1) == 104, "Both files must hold 104 matches"
assert (v1.match_id.values == m.event_id.values).all(), "Row order of lr21_dataset.csv must match kick-off order"

# FIFA ranking published 11 June 2026 (tournament opening day)
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
assert set(fifa_rank) == set(m.home_team) | set(m.away_team), "Team names must match the ranking list"
home_rank = m.home_team.map(fifa_rank)
away_rank = m.away_team.map(fifa_rank)

# Form: goal difference per match in EARLIER matches only (extra time counted, penalties not)
final_h = m.home_score + m.extra_time_score_home.fillna(0)
final_a = m.away_score + m.extra_time_score_away.fillna(0)
history = {t: [] for t in fifa_rank}
form_h, form_a = [], []
for i, r in m.iterrows():
    form_h.append(np.mean(history[r.home_team]) if history[r.home_team] else 0.0)   # before this match
    form_a.append(np.mean(history[r.away_team]) if history[r.away_team] else 0.0)
    g = final_h[i] - final_a[i]                                                     # then record the result
    history[r.home_team].append(g)
    history[r.away_team].append(-g)

rank_gap = away_rank - home_rank
d = pd.DataFrame({
    "match_id": m.event_id, "date": m.date.dt.date, "stage": v1.stage,
    "home_team": m.home_team, "away_team": m.away_team,
    # 8 explanatory variables
    "rank_gap_sqrt": (np.sign(rank_gap) * np.sqrt(rank_gap.abs())).round(4),     # strength, diminishing returns
    "log_rank_ratio": (np.log(away_rank) - np.log(home_rank)).round(4),          # relative strength
    "form_gd_diff": (pd.Series(form_h) - pd.Series(form_a)).round(4),            # earlier GD per match
    "rest_days_diff": v1.rest_days_diff,                                         # schedule
    "host_adv": v1.host_adv,                                                     # USA/Mexico/Canada
    "strong_confed_diff": v1.strong_confed_diff,                                 # UEFA/CONMEBOL
    "knockout": m.group_name.isna().astype(int),                                 # stage known in advance
    "top10_diff": (home_rank <= 10).astype(int) - (away_rank <= 10).astype(int), # elite team
    # target
    "goal_diff": v1.goal_diff,
})
d.to_csv(os.path.join(DATA_DIR, "lr21_dataset_v2.csv"), index=False)
print("Saved lr21_dataset_v2.csv")
print(d.head(8).to_string())

X_cols = ["rank_gap_sqrt", "log_rank_ratio", "form_gd_diff", "rest_days_diff",
          "host_adv", "strong_confed_diff", "knockout", "top10_diff"]
OLD_COLS = ["rank_gap", "form_gf_diff", "form_ga_diff", "form_pts_diff",
            "rest_days_diff", "host_adv", "strong_confed_diff", "debutant_diff"]
TARGET = "goal_diff"
X, y = d[X_cols], d[TARGET]

# 2. Validation
section("2. VALIDATION")
first24 = d.index[:24]
checks = pd.DataFrame({
    "Check": ["Row count = 104", "Explanatory variables = 8", "No missing values",
              "Unique match IDs", "First-match form = 0 (no leakage)",
              "Target identical to v1", "Target equals home - away score"],
    "Pass": [len(d) == 104, len(X_cols) == 8, int(d[X_cols + [TARGET]].isna().sum().sum()) == 0,
             d.match_id.is_unique, bool((d.loc[first24, "form_gd_diff"] == 0).all()),
             bool((d.goal_diff == v1.goal_diff).all()),
             bool((d.goal_diff == m.home_score - m.away_score).all())],
})
save_table(checks, "U1_validation_checks", index=False)
if not checks.Pass.all():
    print("\nWARNING: at least one validation check failed. Review the table above.")

# 3. Exploratory data analysis
section("3. EDA")
desc = d[X_cols + [TARGET]].describe().T[["mean", "std", "min", "50%", "max"]].round(3)
save_table(desc, "U2_descriptive_statistics")

corr = d[X_cols + [TARGET]].corr()
save_table(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3).to_frame("r with goal_diff"),
           "U3_target_correlations")

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True)
plt.title("Correlation matrix: improved features")
save_fig("G1_correlation_heatmap")

# Why the square root helps: raw vs sqrt rank gap
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, col, lab in [(ax[0], v1.rank_gap, "Raw rank gap"), (ax[1], d.rank_gap_sqrt, "Signed sqrt of rank gap")]:
    sns.regplot(x=col, y=y, ax=a, scatter_kws={"alpha": .6, "s": 25}, line_kws={"color": "#C73E1D"})
    a.set(title=f"{lab} (r = {col.corr(y):.3f})", xlabel=lab, ylabel="Goal difference")
save_fig("G2_raw_vs_sqrt_rank_gap")

X_const = sm.add_constant(X)
vif = pd.DataFrame({"Variable": X_cols,
                    "VIF": [variance_inflation_factor(X_const.values, i + 1) for i in range(len(X_cols))]}).round(2)
save_table(vif, "U4_vif", index=False)
print("Note: rank_gap_sqrt and log_rank_ratio overlap (high VIF). Ridge handles this by shrinking both.")

# 4. Models: OLS (for interpretation) and Ridge (for prediction)
section("4. MODELS")
ols = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")   # robust standard errors
print(ols.summary())

ridge = make_ridge().fit(X, y)
alpha = ridge[-1].alpha_
scale = ridge[0].scale_
ridge_coef = pd.Series(ridge[-1].coef_ / scale, index=X_cols)          # back to original units
ridge_int = ridge[-1].intercept_ - (ridge_coef * ridge[0].mean_).sum()
coef = pd.DataFrame({
    "OLS coef": ols.params.drop("const"),
    "OLS robust p": ols.pvalues.drop("const"),
    "Ridge coef": ridge_coef,
    "Ridge std. coef": pd.Series(ridge[-1].coef_ / y.std(ddof=0), index=X_cols),
}).round(4)
save_table(coef, "U5_coefficients")
print(f"\nRidge alpha chosen by CV: {alpha:.2f}; Ridge intercept: {ridge_int:.3f}")

# 5. Evaluation: R2, Adj R2, MAE, MSE, RMSE
section("5. EVALUATION")


def metrics(y_true, y_pred, p=8):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # outcome accuracy: predict a draw when |prediction| < 0.5
    pred_out = np.where(np.abs(y_pred) < 0.5, 0, np.sign(y_pred))
    return {"R2": r2, "Adj R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
            "MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse),
            "Outcome acc": (pred_out == np.sign(y_true)).mean()}


# v1 columns kept alongside v2 so both versions get the same test
d_all = pd.concat([d, v1[["rank_gap", "form_gf_diff", "form_ga_diff", "form_pts_diff", "debutant_diff"]]], axis=1)


def repeated_cv(model, cols, reps=20):
    # 5-fold CV repeated 20 times; every match is predicted by a model that never saw it
    rows = [metrics(y, cross_val_predict(model, d_all[cols], y, cv=KFold(5, shuffle=True, random_state=s)), p=len(cols))
            for s in range(reps)]
    out = pd.DataFrame(rows).mean()
    out["Adj R2"] = np.nan          # in-sample measure only
    return out


# 80/20 train/test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_split = make_ridge().fit(X_tr, y_tr)

ev = pd.DataFrame({
    "Ridge - full sample (n=104)": metrics(y, ridge.predict(X)),
    "Ridge - train 80% (n=83)": metrics(y_tr, ridge_split.predict(X_tr)),
    "Ridge - test 20% (n=21)": metrics(y_te, ridge_split.predict(X_te)),
    "Ridge - 5-fold CV x20": repeated_cv(make_ridge(), X_cols),
    "Baseline - predict mean": metrics(y, np.full(len(y), y.mean())),
}).T.round(3)
ev.loc[["Ridge - test 20% (n=21)", "Baseline - predict mean"], "Adj R2"] = np.nan
save_table(ev, "U6_evaluation_metrics")

# Version 1 vs version 2, same honest test (repeated 5-fold CV)
cmp = pd.DataFrame({
    "V1: original 8, OLS": repeated_cv(LinearRegression(), OLD_COLS),
    "V1 features + Ridge": repeated_cv(make_ridge(), OLD_COLS),
    "rank_gap only, OLS": repeated_cv(LinearRegression(), ["rank_gap"]),
    "V2: improved 8, OLS": repeated_cv(LinearRegression(), X_cols),
    "V2: improved 8, Ridge (FINAL)": repeated_cv(make_ridge(), X_cols),
}).T[["R2", "MAE", "MSE", "RMSE", "Outcome acc"]].round(3)
save_table(cmp, "U7_version_comparison")
gain = cmp.loc["V2: improved 8, Ridge (FINAL)", "R2"] / cmp.loc["V1: original 8, OLS", "R2"] - 1
print(f"\nOut-of-sample R2 improvement over v1: {gain:+.0%}")

# Comparison chart
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
cmp["R2"].plot.barh(ax=ax[0], color=["#9BA3AB"] * 4 + ["#C73E1D"])
ax[0].set(title="Cross-validated R² (higher is better)", xlabel="R²")
cmp["MAE"].plot.barh(ax=ax[1], color=["#9BA3AB"] * 4 + ["#C73E1D"])
ax[1].set(title="Cross-validated MAE in goals (lower is better)", xlabel="MAE")
ax[1].set_xlim(1.0, cmp["MAE"].max() + 0.1)
for a in ax:
    a.invert_yaxis()
    for c in a.containers:
        a.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
save_fig("G3_version_comparison")

# Actual vs predicted (out-of-fold, so it shows real accuracy)
oof = cross_val_predict(make_ridge(), X, y, cv=KFold(5, shuffle=True, random_state=42))
plt.figure(figsize=(6.5, 6))
plt.scatter(oof, y, alpha=.6, color="#2E86AB")
plt.plot([-5, 7], [-5, 7], "r--")
plt.xlim(-5, 7); plt.ylim(-5, 7)
plt.title(f"Actual vs out-of-fold predicted GD (R² = {r2_score(y, oof):.3f})")
plt.xlabel("Predicted goal difference"); plt.ylabel("Actual goal difference")
save_fig("G4_actual_vs_predicted_oof")

# Residual diagnostics and assumption tests (OLS on new features)
resid = ols.resid
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(ols.fittedvalues, resid, alpha=.6); ax[0].axhline(0, color="r", ls="--")
ax[0].set(title="Residuals vs fitted", xlabel="Fitted GD", ylabel="Residual")
stats.probplot(resid, dist="norm", plot=ax[1]); ax[1].set_title("Normal Q-Q plot")
save_fig("G5_residual_diagnostics")
bp = het_breuschpagan(resid, sm.add_constant(X))
sw = stats.shapiro(resid)
print(f"\nBreusch-Pagan p = {bp[1]:.3f} | Shapiro-Wilk p = {sw.pvalue:.3f} | Durbin-Watson = {sm.stats.durbin_watson(resid):.3f}")

# 6. Significance test: is version 2 really better than version 1?
section("6. SIGNIFICANCE TEST")


def avg_oof(model, cols, reps=20):
    # average out-of-fold prediction for each match across 20 CV repeats
    return np.mean([cross_val_predict(model, d_all[cols], y, cv=KFold(5, shuffle=True, random_state=s))
                    for s in range(reps)], axis=0)


err_v1 = (y - avg_oof(LinearRegression(), OLD_COLS)) ** 2
err_v2 = (y - avg_oof(make_ridge(), X_cols)) ** 2
t_test = stats.ttest_rel(err_v1, err_v2)
w_test = stats.wilcoxon(err_v1, err_v2)
print(f"Mean squared error per match: v1 {err_v1.mean():.3f} -> v2 {err_v2.mean():.3f}")
print(f"v2 better in {(err_v1 > err_v2).sum()} of 104 matches")
print(f"Paired t-test p = {t_test.pvalue:.3f} | Wilcoxon p = {w_test.pvalue:.3f}")
print("Significant at 5%" if max(t_test.pvalue, w_test.pvalue) < 0.05
      else "Not significant at 5%: consistent gain, but 104 matches cannot prove it")

# 7. Interpretation
section("7. INTERPRETATION")
eq = " ".join([f"{ridge_int:.3f}"] + [f"{'+' if ridge_coef[c] >= 0 else '-'} {abs(ridge_coef[c]):.4f}*{c}" for c in X_cols])
print(f"Final Ridge model:\nPredicted GD = {eq}")
mex = d[(d.home_team == "Mexico") & (d.away_team == "South Africa")].index[0]
print(f"\nExample: Mexico v South Africa -> predicted {ridge.predict(X.loc[[mex]])[0]:+.2f}, actual {y[mex]:+d}")
print("\nDone. Figures in ./figures, tables in ./tables")