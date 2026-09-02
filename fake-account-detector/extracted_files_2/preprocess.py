"""
Step 2: Preprocess raw data
----------------------------
Raw dataset (profile + behavioral + graph features, imbalanced)
        -> Clean (missing values, duplicates, outliers)
        -> Encode (categorical -> numeric, already mostly numeric here)
        -> Scale (standardize numeric ranges)
        -> Balance (SMOTE oversampling for the minority 'fake' class)
        -> Preprocessed dataset, ready for step 4 (train base models)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

RAW_PATH = "/home/claude/dataset/instagram_fake_account_dataset.csv"
OUT_DIR = "/home/claude/dataset"

df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")
print(f"Class balance (raw): \n{df['is_fake'].value_counts(normalize=True).round(3)}")

# ---------- 1. CLEAN ----------
# Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"\nDropped {before - len(df)} duplicate rows")

# Check missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values found, imputing with median (numeric) / mode (categorical)")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
else:
    print("No missing values found")

# Cap extreme outliers (winsorize at 1st/99th percentile) on skewed count features
skewed_cols = ["followers_count", "following_count", "posts_count",
                "avg_likes_per_post", "avg_comments_per_post"]
for col in skewed_cols:
    low, high = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(low, high)
print(f"Winsorized outliers in: {skewed_cols}")

# ---------- 2. ENCODE ----------
# account_id is an identifier, not a feature -> drop before modeling but keep a copy
account_ids = df["account_id"]
df_model = df.drop(columns=["account_id"])

# All remaining columns are already numeric (binary flags + continuous) in this dataset.
# If real data included categorical fields (e.g. account_type, language), you'd do:
#   df_model = pd.get_dummies(df_model, columns=["account_type", "language"], drop_first=True)
print("\nEncoding: all features already numeric (binary flags + continuous) — no encoding needed here.")
print("(Real-world data with categorical fields would use one-hot or target encoding at this step.)")

# ---------- 3. SPLIT (before scaling/SMOTE to avoid leakage) ----------
X = df_model.drop(columns=["is_fake"])
y = df_model["is_fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain/test split: {X_train.shape[0]} train / {X_test.shape[0]} test")

# ---------- 4. SCALE ----------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
print("Scaled all features to zero mean / unit variance (fit on train only, applied to test)")

# ---------- 5. BALANCE (SMOTE, TRAIN SET ONLY) ----------
print(f"\nClass balance before SMOTE (train): {y_train.value_counts(normalize=True).round(3).to_dict()}")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print(f"Class balance after SMOTE (train): {y_train_bal.value_counts(normalize=True).round(3).to_dict()}")
print(f"Train shape after SMOTE: {X_train_bal.shape}")

# ---------- SAVE ----------
train_out = X_train_bal.copy()
train_out["is_fake"] = y_train_bal.values
train_out.to_csv(f"{OUT_DIR}/preprocessed_train.csv", index=False)

test_out = X_test_scaled.copy()
test_out["is_fake"] = y_test.values
test_out.to_csv(f"{OUT_DIR}/preprocessed_test.csv", index=False)

print(f"\nSaved: {OUT_DIR}/preprocessed_train.csv  (balanced, scaled, for training)")
print(f"Saved: {OUT_DIR}/preprocessed_test.csv   (scaled only, untouched balance, for honest evaluation)")
