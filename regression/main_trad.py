# %% Loading Libraries
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Data Loading
df = pd.read_csv("../databases/Video_Games_Sales_as_at_22_Dec_2016.csv")

# %% Preprocessing
df.drop(columns=["Other_Sales", "Name", "Critic_Count", "Developer", "Rating", "Publisher", "User_Count", "User_Score", "NA_Sales", "JP_Sales", "EU_Sales"], inplace=True, errors='ignore')
df.dropna(subset=["Year_of_Release", "Genre", "Global_Sales", "Critic_Score"], inplace=True)
print(df.head())

# %% Handle outliers
df = df[(np.abs(df["Global_Sales"] - df["Global_Sales"].mean()) / df["Global_Sales"].std() < 3)]
df = df[(np.abs(df["Critic_Score"] - df["Critic_Score"].mean()) / df["Critic_Score"].std() < 3)]


# %% Label encoding for categorical columns
cat_columns = ["Platform", "Genre", "Year_of_Release"]
label_encoders = {}
for col in cat_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# %% Splitting features and target
X = df.drop(columns="Global_Sales")
y = df["Global_Sales"]

print(X[:5])

# %% Scale numeric features (if needed for linear models)
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = np.log1p(df["Global_Sales"])

print(X[:5])

# %% Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_preds = rf_model.predict(X_test)

# Evaluation
rf_mse = mean_squared_error(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("Random Forest Results:")
print("MSE:", rf_mse)
print("RMSE:", np.sqrt(rf_mse))
print("MAE:", rf_mae)
print("R^2 Score:", rf_r2)

# %% XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_preds = xgb_model.predict(X_test)

# Evaluation
xgb_mse = mean_squared_error(y_test, xgb_preds)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

print("XGBoost Results:")
print("MSE:", xgb_mse)
print("RMSE:", np.sqrt(xgb_mse))
print("MAE:", xgb_mae)
print("R^2 Score:", xgb_r2)
# %% LightGBM Regressor
from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)

# Predictions
lgbm_preds = lgbm_model.predict(X_test)

# Evaluation
lgbm_mse = mean_squared_error(y_test, lgbm_preds)
lgbm_mae = mean_absolute_error(y_test, lgbm_preds)
lgbm_r2 = r2_score(y_test, lgbm_preds)

print("LightGBM Results:")
print("MSE:", lgbm_mse)
print("RMSE:", np.sqrt(lgbm_mse))
print("MAE:", lgbm_mae)
print("R^2 Score:", lgbm_r2)
# %% CatBoost Regressor
from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)
cat_model.fit(X_train, y_train)

# Predictions
cat_preds = cat_model.predict(X_test)

# Evaluation
cat_mse = mean_squared_error(y_test, cat_preds)
cat_mae = mean_absolute_error(y_test, cat_preds)
cat_r2 = r2_score(y_test, cat_preds)

print("CatBoost Results:")
print("MSE:", cat_mse)
print("RMSE:", np.sqrt(cat_mse))
print("MAE:", cat_mae)
print("R^2 Score:", cat_r2)

# %% Plotting
def plot_preds(y_true, y_preds, title):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_preds)
    plt.xlabel("Actuals")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.show()

plot_preds(y_test, rf_preds, "Random Forest Predictions")
plot_preds(y_test, xgb_preds, "XGBoost Predictions")
plot_preds(y_test, lgbm_preds, "LightGBM Predictions")
plot_preds(y_test, cat_preds, "CatBoost Predictions")
