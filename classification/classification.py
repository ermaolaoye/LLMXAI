# %% Importing Libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% Decide the Backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device: ", device)

# %% Load the Data
df = pd.read_csv("../databases/bank-full.csv", delimiter=";")
print(df.head())
print(df.value_counts("poutcome"))

# %% Preprocessing
# Get rid of the previous marketing campaign data
df.drop(columns=["default", "contact", "day", "month", "duration", "pdays", "previous", "poutcome"], inplace=True, errors='ignore')
print(df.head())

# %% Encoding
label_encoders = {}
cat_columns = ["job", "marital", "education", "housing", "loan", "y"]
for col in cat_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# %% Splitting data
X = df.drop(columns="y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_test))

# %% Scaling

# %% Creating PyTorch dataset
class BankMarketingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BankMarketingDataset(X_train, y_train)
test_dataset = BankMarketingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

# %% Define the Model
class BankMarketingModel(nn.Module):
    def __init__(self, input_size. hidden_units=[64, 32]):

        super(BankMarketingModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)
