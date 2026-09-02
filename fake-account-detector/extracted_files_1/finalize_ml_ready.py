"""
Final step: engineered features -> ML-ready dataset
------------------------------------------------------
Input: engineered_features.csv (raw + engineered columns, still unscaled)
Output: fully ML-ready train/test sets (encoded, scaled, split, balanced)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("/home/claude/dataset/engineered_features.csv")
print(f"Input (engineered): {df.shape}")

# Drop identifier -- not a feature
df_model = df.drop(columns=["account_id"])

# Encode: all numeric already (no categorical text columns in this dataset)
# Real data with categoricals: df_model = pd.get_dummies(df_model, columns=[...])

X = df_model.drop(columns=["is_fake"])
y = df_model["is_fake"]

# Split BEFORE scaling/SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale (fit on train only)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Balance (train only)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

train_final = X_train_bal.copy()
train_final["is_fake"] = y_train_bal.values
test_final = X_test_scaled.copy()
test_final["is_fake"] = y_test.values

train_final.to_csv("/home/claude/dataset/ml_ready_train.csv", index=False)
test_final.to_csv("/home/claude/dataset/ml_ready_test.csv", index=False)

print(f"Final feature count: {X.shape[1]} (raw + engineered, encoded, scaled)")
print(f"Train: {train_final.shape} (balanced 50/50)")
print(f"Test:  {test_final.shape} (natural {y_test.value_counts(normalize=True)[1]*100:.0f}% fake ratio, untouched)")
print("\nReady for step 4: train base models (LR, DT, RF, XGBoost, ANN)")
