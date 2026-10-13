"""
Step 2 - Fit and evaluate the two linear regression models (OLS).

    LR 2.1  goal_difference  ~ 8 match-level variables      (104 rows)
    LR 2.2  goals_scored     ~ 8 team-level variables       (208 rows)

For each model the script produces
    * the OLS fit (coefficients, standard errors, t-tests, 95% CI, F-test, R-squared, adjusted R-squared)
    * VIF (multicollinearity) and standardised coefficients (to compare variable importance)
    * residual diagnostics (Breusch-Pagan, Shapiro-Wilk, Durbin-Watson, residual plots)
    * out-of-sample checks: repeated 10-fold cross-validation against a "predict the mean" baseline,
      and a chronological hold-out (fit on the 72 group matches, test on the 32 knockout matches)
    * a sensitivity check that adds extra-time goals to the response
and writes tables, figures and a plain results summary into outputs/.

LR 2.2 has two rows per match (one per team) that share the match, so its standard errors are clustered by
match and its cross-validation keeps both rows of a match in the same fold.

Run after step 1:   python src/02_fit_and_evaluate.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "processed", ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# ---- chart styling: one blue for marks, a blue-grey-red diverging scale for correlations --------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})

MODELS = {
    "lr21": dict(
        file="lr21_dataset.csv", y="goal_difference", title="LR 2.1 - goal difference (home - away)", cluster=None,
        x=["xg_net_diff", "finishing_diff", "keeper_goals_prevented_diff", "final_third_entries_diff",
           "fair_play_points_lost_diff", "travel_km_diff", "rest_days_diff", "host_advantage"]),
    "lr22": dict(
        file="lr22_dataset.csv", y="goals_scored", title="LR 2.2 - goals scored by a team", cluster="match_id",
        x=["team_xg_for_avg", "opp_xg_against_avg", "opp_keeper_goals_prevented_avg", "team_box_touches_avg",
           "opp_shots_on_target_faced_avg", "temperature_c", "knockout_stage", "team_is_host"]),
}


# --------------------------------------------------------------------------------------------------
def repeated_cv(X, y, groups=None, repeats=30, folds=10):
    """Repeated k-fold CV. Returns CV RMSE/MAE/R2 of the model and RMSE/MAE of the 'predict the training mean' baseline."""
    res = {k: [] for k in ("rmse", "mae", "r2", "base_rmse", "base_mae")}
    for rep in range(repeats):
        pred, base = np.zeros(len(y)), np.zeros(len(y))
        if groups is None:
            splits = KFold(folds, shuffle=True, random_state=rep).split(X)
        else:   # shuffle the matches reproducibly, then keep both rows of a match together
            rng = np.random.RandomState(rep)
            ids = np.unique(groups)
            order = dict(zip(ids, rng.permutation(len(ids))))
            splits = GroupKFold(folds).split(X, y, np.array([order[g] for g in groups]))
        for train, test in splits:
            pred[test] = LinearRegression().fit(X[train], y[train]).predict(X[test])
            base[test] = y[train].mean()
        res["rmse"].append(np.sqrt(mean_squared_error(y, pred)))
        res["mae"].append(mean_absolute_error(y, pred))
        res["r2"].append(r2_score(y, pred))
        res["base_rmse"].append(np.sqrt(mean_squared_error(y, base)))
        res["base_mae"].append(mean_absolute_error(y, base))
    return {k: float(np.mean(v)) for k, v in res.items()}


def analyse(key, cfg):
    df = pd.read_csv(DATA / cfg["file"])
    x, y = cfg["x"], cfg["y"]
    assert len(x) == 8
    X = sm.add_constant(df[x].astype(float))

    # 1) the fit ------------------------------------------------------------------------------------
    ols = sm.OLS(df[y], X).fit()
    if cfg["cluster"]:
        robust = sm.OLS(df[y], X).fit(cov_type="cluster", cov_kwds={"groups": df[cfg["cluster"]]})
    else:
        robust = sm.OLS(df[y], X).fit(cov_type="HC3")
    print("\n" + "=" * 100 + f"\n{cfg['title']}\n" + "=" * 100)
    print(ols.summary().tables[1])

    # 2) coefficient table with VIF and standardised coefficients -------------------------------------
    sd_x, sd_y = df[x].std(), df[y].std()
    ci = ols.conf_int()
    coef = pd.DataFrame({"coef": ols.params, "std_err": ols.bse, "t": ols.tvalues, "p_value": ols.pvalues,
                         "ci_low": ci[0], "ci_high": ci[1], "robust_p": robust.pvalues})
    coef["std_beta"] = coef.coef * sd_x / sd_y
    coef["std_beta_low"] = coef.ci_low * sd_x / sd_y
    coef["std_beta_high"] = coef.ci_high * sd_x / sd_y
    coef["VIF"] = pd.Series({c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"})
    coef.to_csv(OUT / "tables" / f"{key}_coefficients.csv")

    # 3) residual diagnostics -----------------------------------------------------------------------
    bp_p = het_breuschpagan(ols.resid, X)[1]            # H0: constant residual variance
    sw_p = stats.shapiro(ols.resid).pvalue              # H0: residuals are normal
    dw = durbin_watson(ols.resid)                       # ~2 means no first-order autocorrelation

    # 4) out-of-sample checks -----------------------------------------------------------------------
    groups = df[cfg["cluster"]].values if cfg["cluster"] else None
    cv = repeated_cv(df[x].values.astype(float), df[y].values, groups)
    train = df.match_id <= 72                            # group stage (matches 1-72) -> knockouts (73-104)
    m_ho = LinearRegression().fit(df.loc[train, x], df.loc[train, y])
    p_ho = m_ho.predict(df.loc[~train, x])
    y_te = df.loc[~train, y]
    holdout = dict(n_train=int(train.sum()), n_test=int((~train).sum()),
                   rmse=float(np.sqrt(mean_squared_error(y_te, p_ho))), mae=float(mean_absolute_error(y_te, p_ho)),
                   base_rmse=float(np.sqrt(mean_squared_error(y_te, np.full(len(y_te), df.loc[train, y].mean())))),
                   r2=float(r2_score(y_te, p_ho)))

    # 5) sensitivity: add extra-time goals to the response ---------------------------------------------
    et = pd.read_csv(DATA / "extra_time_targets.csv").set_index("match_id")
    if key == "lr21":
        y_et = df.match_id.map(et.home_goals_incl_et - et.away_goals_incl_et)
    else:
        is_home = df.team == pd.read_csv(DATA / "lr21_dataset.csv").set_index("match_id").home_team.reindex(df.match_id).values
        y_et = np.where(is_home, df.match_id.map(et.home_goals_incl_et), df.match_id.map(et.away_goals_incl_et))
    ols_et = sm.OLS(np.asarray(y_et, dtype=float), X).fit()

    summary = dict(
        model=cfg["title"], n=int(ols.nobs), r2=ols.rsquared, adj_r2=ols.rsquared_adj, f=ols.fvalue, f_p=ols.f_pvalue,
        resid_se=float(np.sqrt(ols.mse_resid)), aic=ols.aic, bp_p=bp_p, shapiro_p=sw_p, dw=dw,
        max_vif=float(coef.VIF.max()), cv_rmse=cv["rmse"], cv_mae=cv["mae"], cv_r2=cv["r2"],
        base_rmse=cv["base_rmse"], base_mae=cv["base_mae"], holdout=holdout,
        sens_incl_et=dict(r2=ols_et.rsquared, adj_r2=ols_et.rsquared_adj),
        robust_type="clustered by match" if cfg["cluster"] else "HC3")
    summary = json.loads(json.dumps(summary, default=float))
    (OUT / "tables" / f"{key}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # 6) figures ------------------------------------------------------------------------------------
    make_figures(key, cfg, df, ols, coef)
    return df, coef, summary


# --------------------------------------------------------------------------------------------------
def make_figures(key, cfg, df, ols, coef):
    y = df[cfg["y"]]
    short = cfg["title"].split(" - ")[0]

    # residual diagnostics: residuals vs fitted | normal Q-Q | actual vs predicted
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax[0].scatter(ols.fittedvalues, ols.resid, s=14, color=BLUE, alpha=0.7, edgecolor=SURFACE, linewidth=0.5)
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set(xlabel="Fitted value", ylabel="Residual", title="Residuals vs fitted")
    (osm, osr), (slope, intercept, _) = stats.probplot(ols.resid, dist="norm")
    ax[1].scatter(osm, osr, s=14, color=BLUE, alpha=0.7, edgecolor=SURFACE, linewidth=0.5)
    ax[1].plot(osm, slope * osm + intercept, color=INK2, lw=0.9)
    ax[1].set(xlabel="Theoretical quantile", ylabel="Residual", title="Normal Q-Q")
    ax[2].scatter(y, ols.fittedvalues, s=14, color=BLUE, alpha=0.7, edgecolor=SURFACE, linewidth=0.5)
    lim = [min(y.min(), ols.fittedvalues.min()), max(y.max(), ols.fittedvalues.max())]
    ax[2].plot(lim, lim, color=INK2, lw=0.9, ls="--")
    ax[2].set(xlabel="Actual", ylabel="Predicted", title="Actual vs predicted")
    fig.suptitle(cfg["title"], x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "figures" / f"{key}_diagnostics.png", dpi=220)
    plt.close(fig)

    # standardised coefficients with 95% CI (comparable across variables)
    c = coef.drop(index="const").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    yy = np.arange(len(c))
    ax.hlines(yy, c.std_beta_low, c.std_beta_high, color=BLUE, lw=2, alpha=0.55)
    ax.scatter(c.std_beta, yy, s=34, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(yy, c.index)
    ax.set(xlabel="Standardised coefficient (95% CI)",
           title=f"{short}: which variables matter")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{key}_coefficients.png", dpi=220)
    plt.close(fig)

    # correlation heatmap (diverging: blue = negative, grey = 0, red = positive)
    cols = cfg["x"] + [cfg["y"]]
    corr = df[cols].corr()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", RED])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else INK)
    ax.grid(False)
    ax.set_title(f"{short}: correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{key}_correlation.png", dpi=220)
    plt.close(fig)
    corr.round(3).to_csv(OUT / "tables" / f"{key}_correlation.csv")


# --------------------------------------------------------------------------------------------------
def md_table(df, floatfmt="{:.3f}"):
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def write_results_md(results):
    lines = ["# Results summary (numbers only)", "", "Generated by `src/02_fit_and_evaluate.py`. "
             "Write the interpretation in your own words.", ""]
    for key, (df, coef, s) in results.items():
        cfg = MODELS[key]
        c = coef.reset_index().rename(columns={"index": "term"})
        c = c[["term", "coef", "std_err", "t", "p_value", "ci_low", "ci_high", "robust_p", "std_beta", "VIF"]]
        c["VIF"] = c.VIF.astype(float)
        h = s["holdout"]
        fit = pd.DataFrame({
            "statistic": ["n", "R-squared", "Adjusted R-squared", "F-statistic", "F-test p-value", "Residual std error",
                          "AIC", "Max VIF", "Breusch-Pagan p", "Shapiro-Wilk p", "Durbin-Watson",
                          "10-fold CV RMSE (30 repeats)", "10-fold CV MAE", "10-fold CV R-squared",
                          "Baseline RMSE (training mean)", "Baseline MAE",
                          f"Hold-out RMSE (train {h['n_train']} -> test {h['n_test']} rows)",
                          "Hold-out baseline RMSE", "Hold-out R-squared",
                          "R-squared with extra-time goals added", "Robust standard errors"],
            "value": [s["n"], s["r2"], s["adj_r2"], s["f"], s["f_p"], s["resid_se"], s["aic"], s["max_vif"],
                      s["bp_p"], s["shapiro_p"], s["dw"], s["cv_rmse"], s["cv_mae"], s["cv_r2"], s["base_rmse"],
                      s["base_mae"], h["rmse"], h["base_rmse"], h["r2"], s["sens_incl_et"]["r2"], s["robust_type"]]})
        fit["value"] = fit.value.apply(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v))
        desc = df[cfg["x"] + [cfg["y"]]].describe().T[["mean", "std", "min", "max"]].reset_index().rename(columns={"index": "variable"})
        lines += [f"## {cfg['title']}", "", "### Descriptive statistics", "", md_table(desc), "",
                  "### Coefficients (OLS)", "", md_table(c), "", "### Fit and validation", "", md_table(fit), ""]
    (OUT / "results_summary.md").write_text("\n".join(lines))


def main():
    results = {k: analyse(k, cfg) for k, cfg in MODELS.items()}
    write_results_md({k: v for k, v in results.items()})
    shared = set(MODELS["lr21"]["x"]) & set(MODELS["lr22"]["x"])
    print("\nColumns with the same name in both models:", sorted(shared) or "none")


if __name__ == "__main__":
    main()
