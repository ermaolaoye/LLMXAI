# %% Importing Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


# %% Load the data
df = pd.read_csv("../databases/bank-full.csv", sep=";")

# %% Preprocessing
# df.drop(columns=["default", "contact", "day", "month",
#                  "duration", "pdays", "previous",
#                  "poutcome"], inplace=True, errors='ignore')

# df = df[(df != "unknown").all(axis=1)]

# print(df.head())

# %% Handleing outliers
df = df[(np.abs(df["balance"] - df["balance"].mean()) / df["balance"].std() < 3)]
df = df[(np.abs(df["age"] - df["age"].mean()) / df["age"].std() < 3)]
df = df[(np.abs(df["duration"] - df["duration"].mean()) / df["duration"].std() < 3)]
df = df[(np.abs(df["campaign"] - df["campaign"].mean()) / df["campaign"].std() < 3)]


# %% Encode categorical variables
df.select_dtypes(exclude=['number']).columns.values

le = LabelEncoder()

df['y'] = le.fit_transform(df['y'])

df = pd.get_dummies(df)

# %% Split into features X and target y
X = df.drop(columns="y")
y = df["y"]

 # %% SMOTE for imbalanced data
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=123)

# X_resampled, y_resampled = smote.fit_resample(X, y)

# resampled_data = pd.concat([X_resampled, pd.Series(y_resampled, name = 'y')], axis=1)

# X = resampled_data.drop("y", axis=1)
# y = resampled_data["y"]


# %% Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %% Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %% Baseline Random Forest

from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest with class_weight='balanced' to handle imbalance
rf_clf = RandomForestClassifier(n_estimators=100,
                                random_state=42,
                                class_weight='balanced')

# Fit on training data
rf_clf.fit(X_train, y_train)

# Predict on test data
y_pred = rf_clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("RandomForest (Baseline) results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
# %% Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_pred_lr))

# %% XGBoost

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    scale_pos_weight=(len(y_train)-sum(y_train)) / sum(y_train),  # analogous to pos_weight
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

print("XGBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("F1:", f1_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_xgb))
# %% LightGBM

import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    scale_pos_weight=(len(y_train)-sum(y_train)) / sum(y_train),
    random_state=42
)

lgb_clf.fit(X_train, y_train)
y_pred_lgb = lgb_clf.predict(X_test)

print("LightGBM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("F1:", f1_score(y_test, y_pred_lgb))
print("Recall:", recall_score(y_test, y_pred_lgb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_lgb))

# %% CatBoost

from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(
    scale_pos_weight=(len(y_train)-sum(y_train)) / sum(y_train),
    random_state=42
)

cat_clf.fit(X_train, y_train)
y_pred_cat = cat_clf.predict(X_test)

print("CatBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_cat))
print("F1:", f1_score(y_test, y_pred_cat))
print("Recall:", recall_score(y_test, y_pred_cat))
print("ROC AUC:", roc_auc_score(y_test, y_pred_cat))

# %% Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, rocauc, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    plt.text(0.5, -0.15, f"ROC AUC: {rocauc:.4f}", ha='center', fontsize=12, transform=plt.gca().transAxes)

    plt.savefig(title + "_confusion_matrix.png")
    plt.show()

plot_confusion_matrix(y_test, y_pred, roc_auc, "RandomForest")
plot_confusion_matrix(y_test, y_pred_lr, roc_auc_score(y_test, y_pred_lr), "LogisticRegression")
plot_confusion_matrix(y_test, y_pred_xgb, roc_auc_score(y_test, y_pred_xgb), "XGBoost")
plot_confusion_matrix(y_test, y_pred_lgb, roc_auc_score(y_test, y_pred_lgb), "LightGBM")
plot_confusion_matrix(y_test, y_pred_cat, roc_auc_score(y_test, y_pred_cat), "CatBoost")
