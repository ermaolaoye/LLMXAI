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

# %% Decide the Backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using Device:", device)

# %% Load the Data
df = pd.read_csv("../databases/bank-full.csv", delimiter=";")
print(df.head())
print(df.value_counts("poutcome"))

# %% Preprocessing
# Get rid of the previous marketing campaign data
df.drop(columns=["default", "contact", "day", "month", "duration", "pdays", "previous", "poutcome"], inplace=True, errors='ignore')
# Get rid of the unknown values
df = df[(df != "unknown").all(axis=1)]
print(df.head())
print(df.value_counts("y"))

# %% Encoding
label_encoders = {}
cat_columns = ["job", "marital", "education", "housing", "loan", "y"]
for col in cat_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


# %% Scaling
numeric_features = ['age', 'balance', 'campaign']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# %% Splitting data
X = df.drop(columns="y", axis=1)
y = df["y"]

# For the test set extract 20% of the data, but ensure that the class distribution is the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(len(X_train), len(X_test))
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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)

# %% Define the Model
class BankMarketingModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[32, 8]):
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
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).to(device))
criterion = FocalLoss(alpha=0.1, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
        preds = (probs >= 0.33).float()

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
cm = confusion_matrix(all_targets, all_preds, normalize='true')
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title("Neural Network")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.text(0.5, -0.15, f"ROC AUC: {roc_auc:.4f}", ha='center', fontsize=12, transform=plt.gca().transAxes)

plt.savefig("nn_confusion_matrix.png")
plt.show()

# %% All hidden layers
# from itertools import product

# hidden_units = [32, 64, 128, 256]
# max_layers = 5

# possible_archs = []

# for num_layers in range(1, max_layers + 1):
#     # Generate all combinations of `num_layers` hidden units
#     for arch in product(hidden_units, repeat=num_layers):
#         possible_archs.append(list(arch))
# print(possible_archs[75])

# # %% Finding the best hyperparameters using Optuna
# import optuna

# def objective(trial):
#     lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
#     dropout = trial.suggest_float("dropout", 0.0, 0.5)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
#     hidden_unit_idx = trial.suggest_int("hidden_units", 0, len(possible_archs) - 1)


#     layers = []
#     input_dim = X_train.shape[1]

#     prev_dim = input_dim
#     for units in possible_archs[hidden_unit_idx]:
#         layers.append(nn.Linear(prev_dim, units))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(p=dropout))
#         prev_dim = units
#     layers.append(nn.Linear(prev_dim, 1))

#     model = nn.Sequential(*layers).to(device)

#     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val).to(device))

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#     batch_size = trial.suggest_int("batch_size", 64, 512, step=64)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#     n_epochs = 100

#     for epoch in range(n_epochs):
#         model.train()
#         total_train_loss = 0.0

#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)

#             optimizer.zero_grad()

#             logits = model(X_batch)
#             loss = criterion(logits, y_batch.unsqueeze(1))

#             loss.backward()
#             optimizer.step()

#         model.eval()
#         all_preds = []
#         all_targets = []

#         with torch.no_grad():
#             for X_batch, y_batch in test_loader:
#                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)

#                 logits = model(X_batch)
#                 probs = torch.sigmoid(logits.squeeze(1))
#                 preds = (probs >= 0.5).float().cpu().numpy()

#                 all_preds.append(preds)
#                 all_targets.append(y_batch.cpu().numpy())

#         # ROC AUC
#         all_preds = np.concatenate(all_preds)

#         all_targets = np.concatenate(all_targets)
#         roc_auc = roc_auc_score(all_targets, all_preds)

#         trial.report(roc_auc, epoch)

#         return roc_auc

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000)

# print("Number of finished trails:", len(study.trials))
# best_trail = study.best_trial
# print("Best trial f1:", best_trail.value)
# print("Best hyperparameters:", best_trail.params)
