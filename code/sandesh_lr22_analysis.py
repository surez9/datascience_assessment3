import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/lr22_dataset.csv")

TARGET = "goals_scored"
FEATURES = [
    "is_host", "is_knockout", "team_matches_played_before",
    "team_goals_scored_before", "team_goals_conceded_before",
    "opp_goals_scored_before", "opp_goals_conceded_before",
    "team_points_before"
]

# Validation
assert df.shape[0] == 208
assert df["match_id"].nunique() == 104
assert df.groupby("match_id").size().eq(2).all()
assert df.isna().sum().sum() == 0
assert df.duplicated().sum() == 0
assert df.duplicated(["match_id", "team"]).sum() == 0

# VIF
X_vif = sm.add_constant(df[FEATURES])
vif = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
})
print(vif.round(3))

# Grouped train/test split: both teams from a match stay in one split
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(
    gss.split(df[FEATURES], df[TARGET], groups=df["match_id"])
)

model = LinearRegression()
model.fit(df.iloc[train_idx][FEATURES], df.iloc[train_idx][TARGET])
pred = model.predict(df.iloc[test_idx][FEATURES])

n = len(test_idx)
p = len(FEATURES)
r2 = r2_score(df.iloc[test_idx][TARGET], pred)
adjusted_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
mae = mean_absolute_error(df.iloc[test_idx][TARGET], pred)
mse = mean_squared_error(df.iloc[test_idx][TARGET], pred)
rmse = np.sqrt(mse)

print("\nR2:", round(r2,4))
print("Adjusted R2:", round(adjusted_r2,4))
print("MAE:", round(mae,4))
print("MSE:", round(mse,4))
print("RMSE:", round(rmse,4))

print("\nIntercept:", round(model.intercept_,6))
print("\nCoefficients:")
print(pd.Series(model.coef_, index=FEATURES).round(6))

# Grouped 5-fold CV
scores=[]
gkf=GroupKFold(n_splits=5)
for fold,(tr,te) in enumerate(gkf.split(df[FEATURES],df[TARGET],groups=df["match_id"]),1):
    m=LinearRegression().fit(df.iloc[tr][FEATURES],df.iloc[tr][TARGET])
    pp=m.predict(df.iloc[te][FEATURES])
    scores.append([
        fold,
        r2_score(df.iloc[te][TARGET],pp),
        mean_absolute_error(df.iloc[te][TARGET],pp),
        mean_squared_error(df.iloc[te][TARGET],pp),
        np.sqrt(mean_squared_error(df.iloc[te][TARGET],pp))
    ])
print("\nGrouped 5-fold CV:")
print(pd.DataFrame(scores,columns=["fold","R2","MAE","MSE","RMSE"]).round(4))
