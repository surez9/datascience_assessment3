"""
Step 4 - Exploratory data analysis (EDA) figure for the report.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, TAB = ROOT / "data" / "processed", ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, GREY, INK, SURFACE = "#2a78d6", "#8a8f98", "#1f2328", "#fcfcfb"

d21 = pd.read_csv(DATA / "lr21_dataset.csv")
d22 = pd.read_csv(DATA / "lr22_dataset.csv")

fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1), facecolor=SURFACE)
for a in ax:
    a.set_facecolor(SURFACE)
    a.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(color="#e6e8eb", lw=0.6)
    a.set_axisbelow(True)

# (a) distribution of the LR 2.1 response ------------------------------------------------------------------
gd = d21.goal_difference
ax[0].hist(gd, bins=np.arange(gd.min() - 0.5, gd.max() + 1.5, 1), color=BLUE, edgecolor=SURFACE)
ax[0].axvline(gd.mean(), color=INK, ls="--", lw=1)
ax[0].set_title("(a) Goal difference per match", loc="left", fontsize=9.5, fontweight="bold", color=INK)
ax[0].set_xlabel("Home minus away goals", fontsize=8.5, color=INK)
ax[0].set_ylabel("Matches", fontsize=8.5, color=INK)
ax[0].text(0.97, 0.95, f"mean {gd.mean():.2f}\nsd {gd.std():.2f}\ndraws {int((gd == 0).sum())} of {len(gd)}",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=8, color=INK)

# (b) strongest LR 2.1 predictor against the response ------------------------------------------------------
x, y = d21.xg_net_diff, gd
slope, intercept = np.polyfit(x, y, 1)
grid = np.linspace(x.min(), x.max(), 50)
ax[1].scatter(x, y, s=16, color=BLUE, alpha=0.75, edgecolor="none")
ax[1].plot(grid, intercept + slope * grid, color=INK, lw=1.2)
resid = y - (intercept + slope * x)
offsets = [((6, 3), "left"), ((8, -2), "left"), ((-8, -2), "right")]    # keep the three labels apart
for i, (xy, ha) in zip(resid.abs().nlargest(3).index, offsets):
    r = d21.loc[i]
    ax[1].annotate(f"{r.home_team} v {r.away_team}", (x[i], y[i]), xytext=xy, textcoords="offset points", ha=ha, fontsize=7, color=INK)
ax[1].set_ylim(y.min() - 0.7, y.max() + 1.4)
ax[1].set_title("(b) Net-xG advantage vs result", loc="left", fontsize=9.5, fontweight="bold", color=INK)
ax[1].set_xlabel("Home minus away net xG in earlier matches", fontsize=8.5, color=INK)
ax[1].set_ylabel("Goal difference", fontsize=8.5, color=INK)
ax[1].text(0.03, 0.95, f"r = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax[1].transAxes, va="top", fontsize=8, color=INK)

# (c) strongest LR 2.2 predictor, grouped into fifths (rows with real history only) --------------------------------------------------
has_history = d22[d22.opp_xg_against_avg != d22.opp_xg_against_avg.mode()[0]].copy()   # drop first-match rows (they carry the filled-in tournament average)
has_history["quintile"] = pd.qcut(has_history.opp_xg_against_avg, 5, labels=False)
g = has_history.groupby("quintile").agg(x=("opp_xg_against_avg", "mean"), goals=("goals_scored", "mean"),
                                se=("goals_scored", lambda s: s.std() / np.sqrt(len(s))), n=("goals_scored", "size"))
ax[2].errorbar(g.x, g.goals, yerr=g.se, fmt="o-", color=BLUE, lw=1.6, ms=6, capsize=2)
ax[2].set_title("(c) Weaker defences concede more", loc="left", fontsize=9.5, fontweight="bold", color=INK)
ax[2].set_xlabel(f"Opponent's average xG conceded (fifths, n = {len(has_history)} rows)", fontsize=8.5, color=INK)
ax[2].set_ylabel("Mean goals scored (+/- 1 s.e.)", fontsize=8.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "eda_overview.png", dpi=220, facecolor=SURFACE)

# numbers quoted in the report
pd.DataFrame({"statistic": ["goal difference mean", "goal difference sd", "draws", "matches", "corr(net xG diff, goal difference)",
                            "three worst-fitted matches"],
              "value": [round(gd.mean(), 3), round(gd.std(), 3), int((gd == 0).sum()), len(gd), round(np.corrcoef(x, y)[0, 1], 3),
                        "; ".join(f"{d21.home_team[i]} v {d21.away_team[i]} ({y[i]:+d})" for i in resid.abs().nlargest(3).index)]}
             ).to_csv(TAB / "eda_numbers.csv", index=False)
print(pd.read_csv(TAB / "eda_numbers.csv").to_string(index=False))
print(g.round(2))
print("saved outputs/figures/eda_overview.png")
