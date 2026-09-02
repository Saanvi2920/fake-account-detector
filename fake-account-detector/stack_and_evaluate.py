"""
Step 5 + 6: Stack ensemble + Evaluation
-----------------------------------------
Takes the 5 base model predictions and combines them via a Logistic
Regression meta-model (stacking). Then evaluates the ensemble on the
held-out test set with full metrics and SHAP-based explanations.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import shap

MODELS_DIR = "D:/Major project/models"
OUT_DIR = "D:/Major project/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load data ----------
train = pd.read_csv("D:/Major project/extracted_files_1/ml_ready_train.csv")
test = pd.read_csv("D:/Major project/extracted_files_1/ml_ready_test.csv")

feature_cols = [c for c in train.columns if c != "is_fake"]
X_train, y_train = train[feature_cols], train["is_fake"]
X_test, y_test = test[feature_cols], test["is_fake"]

# ---------- Load base models and get out-of-fold predictions ----------
base_model_names = [
    "logistic_regression", "decision_tree",
    "random_forest", "xgboost", "ann"
]

base_models = {}
for name in base_model_names:
    base_models[name] = joblib.load(f"{MODELS_DIR}/{name}.joblib")

print("Generating base-model predictions for stacking...")

train_proba = np.column_stack([
    base_models[name].predict_proba(X_train)[:, 1]
    for name in base_model_names
])

test_proba = np.column_stack([
    base_models[name].predict_proba(X_test)[:, 1]
    for name in base_model_names
])

meta_features_train = pd.DataFrame(train_proba, columns=base_model_names)
meta_features_test = pd.DataFrame(test_proba, columns=base_model_names)

# ---------- Train meta-model (Stacking) ----------
print("Training meta-model (Logistic Regression on base predictions)...")
meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_model.fit(meta_features_train, y_train)

y_pred = meta_model.predict(meta_features_test)
y_proba = meta_model.predict_proba(meta_features_test)[:, 1]

# ---------- Evaluate ensemble ----------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*50}")
print("STACK ENSEMBLE - EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Genuine', 'Fake'])}")

# Save ensemble metrics
ensemble_metrics = {
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "roc_auc": round(auc, 4),
}

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fake"])
disp.plot(cmap="Blues", ax=ax, values_format="d")
ax.set_title("Stack Ensemble - Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/confusion_matrix.png")

# ---------- ROC Curve ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Ensemble (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve - Stack Ensemble", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/roc_curve.png")

# ---------- SHAP explanations ----------
print("\nGenerating SHAP explanations...")
explainer = shap.TreeExplainer(base_models["random_forest"])
shap_values = explainer.shap_values(X_test)

# If binary classification, shap_values is a list; take class-1 (fake)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# Summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test, feature_names=feature_cols, show=False, max_display=20)
plt.title("SHAP Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150)
plt.close()
print(f"Saved: {OUT_DIR}/shap_summary.png")

# ---------- Compare base models vs ensemble ----------
with open(f"{MODELS_DIR}/base_model_results.json") as f:
    base_results = json.load(f)

comparison_rows = []
for name, m in base_results.items():
    comparison_rows.append({"Model": name.replace("_", " ").title(), **m})
comparison_rows.append({"Model": "Stack Ensemble", **ensemble_metrics})

comparison_df = pd.DataFrame(comparison_rows)
print(f"\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# Save ensemble model
joblib.dump(meta_model, f"{MODELS_DIR}/stack_ensemble.joblib")
with open(f"{MODELS_DIR}/ensemble_results.json", "w") as f:
    json.dump(ensemble_metrics, f, indent=2)

print(f"\nAll outputs saved to: {OUT_DIR}/")
print(f"Meta-model saved to: {MODELS_DIR}/stack_ensemble.joblib")
