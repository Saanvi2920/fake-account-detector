"""
Step 3: Engineer features
---------------------------
Takes the CLEANED raw dataset (post step-2 cleaning, pre-scaling) and derives
new features across all four categories in the pipeline: profile, behavior,
text, and graph. New features are combinations/transforms of raw ones that
are more directly predictive than the raw values alone.

Run AFTER cleaning, BEFORE scaling/encoding/SMOTE -- ratios and interaction
terms need real units to be meaningful; computing them on already-scaled
(zero-mean) data would produce nonsense ratios.
"""

import pandas as pd
import numpy as np

RAW_PATH = "/home/claude/dataset/instagram_fake_account_dataset.csv"
OUT_PATH = "/home/claude/dataset/engineered_features.csv"

df = pd.read_csv(RAW_PATH)

# ---------- re-apply cleaning from step 2 (dedupe, outlier caps) ----------
df = df.drop_duplicates()
skewed_cols = ["followers_count", "following_count", "posts_count",
                "avg_likes_per_post", "avg_comments_per_post"]
for col in skewed_cols:
    low, high = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(low, high)

print(f"Starting from cleaned raw data: {df.shape}")

# ===================== PROFILE-DERIVED FEATURES =====================
# Completeness score: how "filled out" is the profile (fakes tend to skip this)
df["profile_completeness_score"] = (
    df["has_profile_pic"] + df["has_external_url"] +
    (df["bio_length"] > 20).astype(int)
) / 3

# Username suspicion score: long + digit-heavy usernames are a bot signature
df["username_suspicion_score"] = (
    (df["username_length"] > 12).astype(int) * 0.5 +
    (df["username_digit_ratio"] > 0.3).astype(int) * 0.5
)

# ===================== BEHAVIORAL-DERIVED FEATURES =====================
# Posts per day of account life -- captures activity relative to account age,
# not just raw counts (a 1-week-old account with 5 posts != a 3-year-old one)
df["posts_per_day_alive"] = df["posts_count"] / df["account_age_days"].replace(0, 1)

# Engagement-to-audience mismatch: real engagement should scale with followers.
# A big gap (high followers, near-zero interaction) is a strong fake signal.
expected_engagement = np.log1p(df["followers_count"]) * 0.05
df["engagement_deficit"] = expected_engagement - df["engagement_rate"]

# Activity irregularity: combine low posting frequency with high night-activity skew
df["bot_activity_pattern"] = df["night_activity_ratio"] * (1 / (df["posting_freq_per_week"] + 0.1))

# Follow-ratio imbalance: accounts that follow far more than they're followed by
# (classic follow-for-follow / spam pattern)
df["aggressive_following_flag"] = (df["follower_following_ratio"] < 0.3).astype(int)

# ===================== TEXT-DERIVED FEATURES (from bio) =====================
# We don't have raw bio text here (only bio_length), but this is where you'd add:
#   - keyword flags (e.g. bio contains "follow back", "dm for promo", emoji spam)
#   - bio language/script consistency vs. username language
#   - sentiment/readability score of bio text
# Proxy using available signal: near-empty bio combined with no profile pic
df["empty_identity_flag"] = ((df["bio_length"] < 10) & (df["has_profile_pic"] == 0)).astype(int)

# ===================== GRAPH-DERIVED FEATURES =====================
# Local density anomaly: high degree but low clustering = follows many accounts
# that don't know each other (bot-ring pattern rather than organic friend group)
df["network_anomaly_score"] = df["graph_degree"] * (1 - df["clustering_coefficient"])

# Hub-likelihood: is this account a structural bridge in the network?
# (real influencers score high; isolated bot accounts score near zero)
df["is_network_hub"] = (df["betweenness_centrality"] > df["betweenness_centrality"].quantile(0.9)).astype(int)

print(f"\nEngineered {9} new features:")
new_cols = [
    "profile_completeness_score", "username_suspicion_score",
    "posts_per_day_alive", "engagement_deficit", "bot_activity_pattern",
    "aggressive_following_flag", "empty_identity_flag",
    "network_anomaly_score", "is_network_hub"
]
for c in new_cols:
    print(f"  - {c}")

# ---------- sanity check: do engineered features actually separate the classes? ----------
print("\nMean value by class (fake vs genuine) -- bigger gaps = more predictive:")
comparison = df.groupby("is_fake")[new_cols].mean().T
comparison.columns = ["genuine_avg", "fake_avg"]
comparison["gap"] = (comparison["fake_avg"] - comparison["genuine_avg"]).abs()
print(comparison.sort_values("gap", ascending=False).round(4).to_string())

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  shape={df.shape}")
print("Next: run this through step-2-style encode/scale/SMOTE before training (step 4).")
