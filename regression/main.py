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
import os
import pandas as pd

# %% Downloading data
path = kh.dataset_download("rush4ratio/video-game-sales-with-ratings")
# %% Importing data
df = pd.read_csv(path + "/Video_Games_Sales_as_at_22_Dec_2016.csv")

print(df.head())

# %% Preprocessing
df.dropna(subset=["Platform", "Year_of_Release", "Genre", "Publisher", "Global_Sales", "Critic_Score", "User_Score"], inplace=True)
df.drop(columns=["Name", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "User_Count", "Critic_Count", "Publisher", "Rating", "Developer"], inplace=True, errors='ignore')

X = df.drop(columns=["Global_Sales"], axis=1)
y = df["Global_Sales"]

# %% Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Encode categorical data and scale numerical data
