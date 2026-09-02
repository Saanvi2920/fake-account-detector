"""
Step 4: Train base models
--------------------------
Trains 5 base classifiers on the ML-ready dataset:
  - Logistic Regression (LR)
  - Decision Tree (DT)
  - Random Forest (RF)
  - XGBoost
  - Artificial Neural Network (ANN)

Outputs per-model metrics and saves each trained model via joblib.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

OUT_DIR = "D:/Major project/models"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load ML-ready data ----------
train = pd.read_csv("D:/Major project/extracted_files_1/ml_ready_train.csv")
test = pd.read_csv("D:/Major project/extracted_files_1/ml_ready_test.csv")
# Copy raw test data for reference
raw_test = pd.read_csv("D:/Major project/extracted_files/instagram_fake_account_dataset.csv")

feature_cols = [c for c in train.columns if c != "is_fake"]
X_train, y_train = train[feature_cols], train["is_fake"]
X_test, y_test = test[feature_cols], test["is_fake"]

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Feature count: {len(feature_cols)}")

# ---------- Define models ----------
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=15, min_samples_split=5, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        n_jobs=-1, random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        n_jobs=-1, random_state=42
    ),
    "ann": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        max_iter=500, early_stopping=True, random_state=42
    ),
}

# ---------- Train, evaluate, save ----------
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    joblib.dump(model, f"{OUT_DIR}/{name}.joblib")

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
    }

# Save results
with open(f"{OUT_DIR}/base_model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*50}")
print("All base models trained and saved.")
print(f"Models saved to: {OUT_DIR}/")
print(f"Results summary:")
for name, metrics in results.items():
    print(f"  {name:25s} | Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")
