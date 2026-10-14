"""
Step 3 - Comparing competing algorithms for both tasks.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression, PoissonRegressor, RidgeCV)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "processed", ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
REPEATS, FOLDS = 20, 10


def algorithms(include_poisson):
    """Name -> function that builds a fresh, untrained model. Penalties are chosen by inner cross-validation."""
    algos = {
        "OLS (linear regression)": lambda: LinearRegression(),
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso": lambda: make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=20000, random_state=0)),
        "Elastic net": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=20000, random_state=0)),
        "Random forest": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=1),
    }
    if include_poisson:                      # goals are counts, so a Poisson model is the natural competitor
        algos["Poisson regression"] = lambda: make_pipeline(
            StandardScaler(), GridSearchCV(PoissonRegressor(max_iter=2000), {"alpha": [0.01, 0.1, 1, 10]}, cv=5))
    return algos


def folds(n, groups, seed):
    if groups is None:
        return KFold(FOLDS, shuffle=True, random_state=seed).split(np.arange(n))
    rng = np.random.RandomState(seed)                       # shuffle the matches, keep both rows of a match together
    ids = np.unique(groups)
    order = dict(zip(ids, rng.permutation(len(ids))))
    return GroupKFold(FOLDS).split(np.arange(n), groups=np.array([order[g] for g in groups]))


def evaluate(X, y, groups, build):
    """Returns RMSE / MAE / R2 for every CV repeat. build=None means the mean-only baseline."""
    rows = []
    for seed in range(REPEATS):
        pred = np.zeros(len(y))
        for tr, te in folds(len(y), groups, seed):
            pred[te] = y[tr].mean() if build is None else build().fit(X[tr], y[tr]).predict(X[te])
        rows.append((np.sqrt(mean_squared_error(y, pred)), mean_absolute_error(y, pred), r2_score(y, pred)))
    return np.array(rows)


def holdout(X, y, train, build):
    tr, te = train, ~train
    pred = np.full(te.sum(), y[tr].mean()) if build is None else build().fit(X[tr], y[tr]).predict(X[te])
    return float(np.sqrt(mean_squared_error(y[te], pred)))


TASKS = {"LR 2.1 - goal difference": ("lr21_dataset.csv", "goal_difference", None, False),
         "LR 2.2 - goals scored": ("lr22_dataset.csv", "goals_scored", "match_id", True)}
tables = {}
for title, (file, target, cluster, poisson) in TASKS.items():
    df = pd.read_csv(DATA / file)
    X, y = df[df.columns[7:15]].values.astype(float), df[target].values.astype(float)
    groups = df[cluster].values if cluster else None
    train = (df.match_id <= 72).values
    rows = []
    for name, build in [("Baseline (mean only)", None)] + list(algorithms(poisson).items()):
        r = evaluate(X, y, groups, build)
        rows.append(dict(algorithm=name, cv_rmse=r[:, 0].mean(), cv_rmse_sd=r[:, 0].std(), cv_mae=r[:, 1].mean(),
                         cv_r2=r[:, 2].mean(), knockout_holdout_rmse=holdout(X, y, train, build)))
        print(title, name, round(rows[-1]["cv_rmse"], 3), flush=True)
    tables[title] = pd.DataFrame(rows).round(3)
    tables[title].to_csv(OUT / "tables" / f"{'lr21' if 'LR 2.1' in title else 'lr22'}_model_comparison.csv", index=False)

# ---- figure: cross-validated RMSE of every algorithm (dot = mean over repeats, bar = +/- 1 sd), baseline dashed ----------
BLUE, GREY, INK, SURFACE = "#2a78d6", "#8a8f98", "#1f2328", "#fcfcfb"
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), facecolor=SURFACE)
for ax, (title, t) in zip(axes, tables.items()):
    t = t.iloc[::-1].reset_index(drop=True)                 # baseline ends up at the bottom
    base = t.loc[t.algorithm.str.startswith("Baseline"), "cv_rmse"].iloc[0]
    ax.set_facecolor(SURFACE)
    ax.axvline(base, color=GREY, ls="--", lw=1)
    for i, r in t.iterrows():
        is_base = r.algorithm.startswith("Baseline")
        ax.errorbar(r.cv_rmse, i, xerr=r.cv_rmse_sd, fmt="o", color=GREY if is_base else BLUE, ms=6, lw=1.6, capsize=0)
        ax.text(r.cv_rmse, i + 0.28, f"{r.cv_rmse:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.algorithm, fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("Cross-validated RMSE (goals; lower is better)", fontsize=8.5, color=INK)
    lo, hi = t.cv_rmse.min() - t.cv_rmse_sd.max() * 3, max(t.cv_rmse.max(), base) + t.cv_rmse_sd.max() * 3
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e8eb", lw=0.6)
fig.tight_layout()
fig.savefig(OUT / "figures" / "model_comparison.png", dpi=220, facecolor=SURFACE)

with open(OUT / "model_comparison.md", "w") as f:
    for title, t in tables.items():
        f.write(f"## {title}\n\n" + t.to_markdown(index=False) + "\n\n")
print("saved outputs/figures/model_comparison.png, outputs/model_comparison.md")
