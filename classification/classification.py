# %% Importing Libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, recall_score

# %% Load the data
df = pd.read_csv("../databases/bank-full.csv", sep=";")

%% Preprocessing
df.drop(columns=["default", "contact", "day", "month",
                 "duration", "pdays", "previous",
                 "poutcome"], inplace=True, errors='ignore')

df = df[(df != "unknown").all(axis=1)]

print(df.head())

# %% Handleing outliers
df = df[(np.abs(df["balance"] - df["balance"].mean()) / df["balance"].std() < 3)]
df = df[(np.abs(df["age"] - df["age"].mean()) / df["age"].std() < 3)]
# df = df[(np.abs(df["duration"] - df["duration"].mean()) / df["duration"].std() < 3)]
df = df[(np.abs(df["campaign"] - df["campaign"].mean()) / df["campaign"].std() < 3)]


# %% Encode categorical variables
df.select_dtypes(exclude=['number']).columns.values

le = LabelEncoder()

df['y'] = le.fit_transform(df['y'])

df = pd.get_dummies(df)

# %% Split into features X and target y
X = df.drop(columns="y")
y = df["y"]

# # %% SMOTE for imbalanced data
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=32)

# X_resampled, y_resampled = smote.fit_resample(X, y)

# resampled_data = pd.concat([X_resampled, pd.Series(y_resampled, name = 'y')], axis=1)

# X = resampled_data.drop("y", axis=1)
# y = resampled_data["y"]


# %% Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %% Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Decide the Backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device:", device)

# %% Creating PyTorch dataset
class BankMarketingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BankMarketingDataset(X_train, y_train)
test_dataset = BankMarketingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# %% Define the Model
class BankMarketingModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[16, 8, 4]):
        super(BankMarketingModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(p=0.2))  # 20% dropout
            prev_dim = units

        layers.append(nn.Linear(prev_dim, 1)) # Single output layer
        self.net = nn.Sequential(*layers)



    def forward(self, x):
        return self.net(x)


# %% Tune the data as negative class is more than positive class
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
pos_weight_val = num_neg / num_pos

# %% Alternate choice of FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha (float): Weighting factor for the rare class (default 1.0 = no weighting).
        gamma (float): Focusing parameter to down-weight easy examples (default 2.0).
        reduction (str): 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw outputs of the model (before sigmoid), shape [batch_size, 1].
        targets: ground truth labels, shape [batch_size, 1] or [batch_size].
        """
        # Compute BCE loss per sample (but not averaged yet)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Probability for positive class
        preds = torch.sigmoid(logits)

        # For stability, clamp predictions in [1e-6, 1-1e-6]
        preds = preds.clamp(min=1e-6, max=1.0 - 1e-6)

        # If targets=1, pt = p, else pt = 1-p
        pt = preds * targets + (1.0 - preds) * (1.0 - targets)

        # Focal term = alpha * (1 - pt)^gamma
        focal_term = self.alpha * (1.0 - pt) ** self.gamma

        focal_loss = focal_term * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# %% Training
model = BankMarketingModel(X_train.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).to(device))
# criterion = FocalLoss(alpha=0.1, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

n_epochs = 10000

train_losses = []
test_losses = []

early_stopping = 20
epochs_no_improve = 0
best_loss = np.inf

for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        # Move the data to the device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item() * len(y_batch)

    epoch_train_loss = total_train_loss / len(train_dataset)

    # Evaluation
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))

            total_test_loss += loss.item() * len(y_batch)

    epoch_val_loss = total_test_loss / len(test_dataset)

    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_train_loss:.4f} - Test Loss: {epoch_val_loss:.4f}")

    # Update the learning rate
    scheduler.step(epoch_val_loss)

    print("Learning rate:", optimizer.param_groups[0]['lr'])

    # Save the Model every 500 epochs
    if (epoch+1) % 500 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    # Early stopping

    if epoch_val_loss < best_loss:
        print(f"Validation loss decreased from {best_loss:.4f} to {epoch_val_loss:.4f}")
        best_loss = epoch_val_loss
        epochs_no_improve = 0
        # Save the model
        torch.save(model.state_dict(), "model_best.pth")

        # Print save model with different color
        print("\033[92m" + "Model saved!" + "\033[0m")
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
plt.ylabel("Focal Loss")
plt.title("Training and Validation Loss")
plt.legend()
# save the plot
plt.savefig("nn_loss_plot.png")
plt.show()

# %% Load the best model
model.load_state_dict(torch.load("model_best.pth", weights_only=True))

# %% Evaluation
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs >= 0.7).float()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

accuracy = accuracy_score(all_targets, all_preds)
print("Accuracy:", accuracy)

# F1 Score
f1 = f1_score(all_targets, all_preds)
print("F1 Score:", f1)
# Recall
recall = recall_score(all_targets, all_preds)
print("Recall:", recall)
# ROC AUC
roc_auc = roc_auc_score(all_targets, all_preds)
print("ROC AUC:", roc_auc)
# Confusion matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title("Neural Network")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.text(0.5, -0.15, f"ROC AUC: {roc_auc:.4f}", ha='center', fontsize=12, transform=plt.gca().transAxes)

plt.savefig("nn_confusion_matrix.png")
plt.show()
