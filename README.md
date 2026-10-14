# FIFA World Cup 2026 - Linear Regression 

| LR 2.1 | 104 (one per match) 
| LR 2.2 | 208 (one per team per match) 

## How to run

```
pip install -r requirements.txt     # install the required libraries
python src/01_prepare_data.py          # builds data/processed/*.csv
python src/02_fit_and_evaluate.py      # fits both models, writes outputs/
python src/03_compare_models.py        # compares competing algorithms (takes about 10-15 minutes)
python src/04_eda.py                   # exploratory figure used in the report
```

Results are written to `outputs/tables`, `outputs/figures`.

## Explanatory variables

LR 2.1:

| Variable | Meaning |
| `xg_net_diff` | average (xG created - xG conceded) in earlier matches |
| `finishing_diff` | average (goals - xG) in earlier matches |
| `keeper_goals_prevented_diff` | average goals prevented by the goalkeeper |
| `final_third_entries_diff` | average final-third entries |
| `fair_play_points_lost_diff` | average yellow cards + 4 x red cards |
| `travel_km_diff` | km from the team's previous venue to this venue |
| `rest_days_diff` | days since the team's previous match |
| `host_advantage` | +1 if the home-listed team is a host playing in its own country, -1 if the away team is, else 0 |

LR 2.2:

| Variable | Meaning |
| `team_xg_for_avg` | team's average xG created in earlier matches |
| `opp_xg_against_avg` | opponent's average xG conceded |
| `opp_keeper_goals_prevented_avg` | opponent goalkeeper's average goals prevented |
| `team_box_touches_avg` | team's average touches in the opposition penalty area |
| `opp_shots_on_target_faced_avg` | average shots on target the opponent has faced |
| `temperature_c` | forecast temperature at the venue (deg C) |
| `knockout_stage` | 1 from the Round of 32 onward |
| `team_is_host` | 1 if the team is a host playing in its own country |


