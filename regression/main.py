# %% Importing libraries
import kagglehub as kh
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Check if GPU is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device: ", device)
# %% Downloading data
path = kh.dataset_download("rush4ratio/video-game-sales-with-ratings")
# %% Importing data
df = pd.read_csv(path + "/Video_Games_Sales_as_at_22_Dec_2016.csv")

print(df.head())

# %% Preprocessing
df.drop(columns=["Other_Sales", "Name", "Critic_Count", "Developer", "Rating", "User_Count", "User_Score", "NA_Sales", "JP_Sales", "EU_Sales"], inplace=True, errors='ignore')
df.dropna(subset=["Year_of_Release", "Genre", "Global_Sales", "Critic_Score"], inplace=True)

print(df.head())

# %% Encoding
label_encoders = {}
cat_columns = ["Platform", "Genre", "Publisher"]
for col in cat_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# %% Splitting data
X = df.drop(columns="Global_Sales", axis=1)
y = df["Global_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(len(X_train), len(X_test))

# %% Scaling
# numeric_columns = ['Critic_Score']
# scaler = StandardScaler()
# X_train[ numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
# X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
# %% Creating PyTorch dataset
class GlobalSalesDataset(Dataset):
    def __init__(self, X, y, cat_cols, num_cols):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        cat_values = self.X.loc[idx, self.cat_cols].values
        cat_data = torch.tensor(cat_values, dtype=torch.long)

        num_values = self.X.loc[idx, self.num_cols].values
        numeric_data = torch.tensor(num_values, dtype=torch.float)
        label = torch.tensor(self.y.iloc[idx], dtype=torch.float)
        return cat_data, numeric_data, label

cat_cols = ["Platform", "Genre"]
num_cols = ["Critic_Score"]
train_dataset = GlobalSalesDataset(X_train, y_train, cat_cols, num_cols)
test_dataset = GlobalSalesDataset(X_test, y_test, cat_cols, num_cols)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)

# %% Creating model
class GlobalSalesModel(nn.Module):
    def __init__(self, cat_cardinalities, embedding_dims, num_numeric, hidden_units=[128,64, 32]):

        super().__init__()

        self.embeddings = nn.ModuleList([nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, embedding_dims)])

        total_emb_dim = sum(embedding_dims)

        # Fully connected layers
        all_input_dim = total_emb_dim + num_numeric

        layers = []
        prev_dim = all_input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.15))
            prev_dim = units

        layers.append(nn.Linear(prev_dim, 1)) # Output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_data, numeric_data):
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(cat_data[:, i]))

        # Concatenating embeddings
        x = torch.cat(embedded, 1)
        # Concatenating numeric data
        x = torch.cat([x, numeric_data], 1)

        # Forward pass
        out = self.mlp(x)
        return out.squeeze(1)

cat_cardinalities = [df[col].nunique() for col in cat_cols] # Number of unique values in each categorical column
embedding_dims = [min(50, (card + 1) // 2) for card in cat_cardinalities] # Embedding dimensions
num_numeric = len(num_cols) # Number of numeric columns

model = GlobalSalesModel(cat_cardinalities, embedding_dims, num_numeric)
model = model.to(device)

# %% Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 1000

scaler = torch.GradScaler()


train_losses = []
test_losses = []

early_stopping = 100
epochs_no_improve = 0
best_loss = np.inf

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for cat_data, numeric_data, labels in train_loader:
        # Move data to device
        cat_data = cat_data.to(device)
        numeric_data = numeric_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if device == torch.device("cuda"):
            with torch.autocast("cuda"):
                out = model(cat_data, numeric_data)
                loss = criterion(out, labels)
        else:
            out = model(cat_data, numeric_data)
            loss = criterion(out, labels)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * len(labels)

    epoch_train_loss = train_loss / len(train_dataset)

    model.eval()
    val_loss = 0
    for cat_data, numeric_data, labels in test_loader:
        cat_data = cat_data.to(device)
        numeric_data = numeric_data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(cat_data, numeric_data)
            loss = criterion(out, labels)

        val_loss += loss.item() * len(labels)

    epoch_val_loss = val_loss / len(test_dataset)

    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Test Loss: {epoch_val_loss:.4f}")

    # Save the Model every 500 epochs
    if (epoch+1) % 500 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    # Early stopping

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping:
        print("Early stopping!")
        # Save the model
        torch.save(model.state_dict(), "model_early_stopping.pth")
        break


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# %% Saving model
torch.save(model.state_dict(), "model.pth")

# %% Eval
model.eval()
preds = []
actuals = []
with torch.no_grad():
    for cat_data, numeric_data, labels in test_loader:
        cat_data = cat_data.to(device)
        numeric_data = numeric_data.to(device)
        labels = labels.to(device)

        out = model(cat_data, numeric_data)
        preds.extend(out.cpu().numpy())
        actuals.extend(labels.cpu().numpy())



mse = mean_squared_error(actuals, preds)
mae = mean_absolute_error(actuals, preds)
r2 = r2_score(actuals, preds)

print("Test MSE:", mse)
print("Test RMSE:", np.sqrt(mse))
print("Test MAE:", mae)
print("R^2 Score:", r2)

plt.figure(figsize=(8,6))
sns.scatterplot(x=actuals, y=preds)
plt.xlabel("Actual total_sales")
plt.ylabel("Predicted total_sales")
plt.title("Predicted vs. Actual total_sales")
plt.show()
