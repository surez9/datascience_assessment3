# FIFA World Cup 2026 — Linear Regression 2.2

**Student responsibility:** Sandesh — modelling and evaluation.

## Final confirmed predictors
1. is_host
2. is_knockout
3. team_matches_played_before
4. team_goals_scored_before
5. team_goals_conceded_before
6. opp_goals_scored_before
7. opp_goals_conceded_before
8. team_points_before

**Target:** goals_scored

## Final test-set evaluation
R² = 0.0204
Adjusted R² = -0.2171
MAE = 1.2723
MSE = 2.5285
RMSE = 1.5901

## Reproducibility
Run:
`python code/sandesh_lr22_analysis.py`

The model uses a grouped 80/20 train-test split with `match_id` as the grouping variable, so both team observations from a match stay in the same split.
