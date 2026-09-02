import os, sys, joblib, pandas as pd, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from auto_report import report_fake_account, get_all_reports

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

print("Testing model loading...")
ensemble = joblib.load(os.path.join(MODELS_DIR, "stack_ensemble.joblib"))
base_models = {
    n: joblib.load(os.path.join(MODELS_DIR, f"{n}.joblib"))
    for n in ["logistic_regression", "decision_tree", "random_forest", "xgboost", "ann"]
}
print(f"Loaded {len(base_models)} base models + ensemble")

print("Testing data loading...")
test = pd.read_csv(os.path.join(DATA_DIR, "ml_ready_test.csv"))
print(f"Test data: {test.shape}")

print("Testing prediction...")
feature_cols = [c for c in test.columns if c != "is_fake"]
X_test = test[feature_cols][:5]
base_probas = np.column_stack([
    base_models[n].predict_proba(X_test)[:, 1] for n in base_models
])
probas = ensemble.predict_proba(base_probas)[:, 1]
labels = ["FAKE" if p >= 0.5 else "GENUINE" for p in probas]
print(f"Sample predictions: {labels}")
print(f"Sample confidence: {[round(p,4) for p in probas]}")

print("Testing auto-report...")
feat = {col: float(test.iloc[0][col]) for col in feature_cols}
report = report_fake_account("acc_test", float(probas[0]), feat, username="test_user")
print(f"Report: {report['report_id']} | Risk: {report['risk_level']}")

print("ALL SYSTEMS WORKING!")
