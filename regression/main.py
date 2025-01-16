# %% Importing libraries
import kagglehub as kh
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd

# %% Downloading data
path = kh.dataset_download("rush4ratio/video-game-sales-with-ratings")
# %% Importing data
df = pd.read_csv(path + "/Video_Games_Sales_as_at_22_Dec_2016.csv")

print(df.head())

# %% Preprocessing
df.replace("tbd", np.nan, inplace=True)
df.dropna(subset=["Platform", "Year_of_Release", "Genre", "Publisher", "Global_Sales", "Critic_Score", "User_Score"], inplace=True)
df.drop(columns=["Name", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "User_Count", "Critic_Count", "Publisher", "Rating", "Developer"], inplace=True, errors='ignore')

X = df.drop(columns=["Global_Sales"], axis=1)
y = df["Global_Sales"]

# %% Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.head())
# %% Encode categorical data and scale numerical data
numeric_features = ["Year_of_Release", "Critic_Score", "User_Score"]
categorical_features = ["Platform", "Genre"]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %% Choosing a Regression Model

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# %% Training the model
model.fit(X_train, y_train)

# %% Evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test R^2:", r2)

# %% Predict the global sales of a new game
new_game = pd.DataFrame({
    "Platform": ["NES"],
    "Year_of_Release": [1980],
    "Genre": ["Action"],
    "Critic_Score": [100],
    "User_Score": [10]
})

new_game_pred = model.predict(new_game)
print("Predicted global sales of the new game:", new_game_pred[0])
