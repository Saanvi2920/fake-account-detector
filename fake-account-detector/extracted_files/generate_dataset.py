"""
Instagram Fake Account Detection - Dataset Generator
------------------------------------------------------
Generates a realistic, labeled dataset combining:
  1. Profile features
  2. Behavioral features
  3. Graph/network features

Feature definitions are modeled after published fake-account-detection
research (e.g. the widely-used Kaggle "Instagram Fake Spammer Genuine
Accounts" dataset) so this can be swapped for real data later without
changing your downstream pipeline (steps 2-9).
"""

import numpy as np
import pandas as pd
import networkx as nx
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N_ACCOUNTS = 5000
FAKE_RATIO = 0.28  # realistic proportion of fake/spam accounts in the wild


def generate_profile_and_behavioral_features(n, fake_ratio):
    n_fake = int(n * fake_ratio)
    n_genuine = n - n_fake
    labels = np.array([1] * n_fake + [0] * n_genuine)  # 1 = fake, 0 = genuine
    np.random.shuffle(labels)

    rows = []
    for i in range(n):
        is_fake = labels[i] == 1

        if is_fake:
            username_len = np.random.randint(8, 20)
            username_digit_ratio = np.random.uniform(0.3, 0.8)
            has_profile_pic = np.random.choice([0, 1], p=[0.6, 0.4])
            bio_length = np.random.randint(0, 15)
            external_url = np.random.choice([0, 1], p=[0.85, 0.15])
            private_account = np.random.choice([0, 1], p=[0.3, 0.7])
            followers = int(np.random.exponential(50))
            following = int(np.random.exponential(800)) + 200
            posts = np.random.poisson(2)
            avg_likes_per_post = np.random.exponential(2)
            avg_comments_per_post = np.random.exponential(0.3)
            posting_freq_per_week = np.random.exponential(0.5)
            account_age_days = np.random.randint(1, 200)
            engagement_rate = (avg_likes_per_post + avg_comments_per_post) / max(followers, 1)
            night_activity_ratio = np.random.uniform(0.4, 0.9)  # bots often post odd hours
        else:
            username_len = np.random.randint(4, 15)
            username_digit_ratio = np.random.uniform(0.0, 0.2)
            has_profile_pic = np.random.choice([0, 1], p=[0.05, 0.95])
            bio_length = np.random.randint(10, 150)
            external_url = np.random.choice([0, 1], p=[0.6, 0.4])
            private_account = np.random.choice([0, 1], p=[0.55, 0.45])
            followers = int(np.random.lognormal(mean=5.5, sigma=1.3))
            following = int(np.random.lognormal(mean=5.0, sigma=1.0))
            posts = np.random.poisson(45)
            avg_likes_per_post = np.random.exponential(35)
            avg_comments_per_post = np.random.exponential(4)
            posting_freq_per_week = np.random.exponential(3)
            account_age_days = np.random.randint(180, 3000)
            engagement_rate = (avg_likes_per_post + avg_comments_per_post) / max(followers, 1)
            night_activity_ratio = np.random.uniform(0.05, 0.35)

        follower_following_ratio = followers / max(following, 1)

        rows.append({
            "account_id": f"acc_{i}",
            "username_length": username_len,
            "username_digit_ratio": round(username_digit_ratio, 3),
            "has_profile_pic": has_profile_pic,
            "bio_length": bio_length,
            "has_external_url": external_url,
            "is_private": private_account,
            "followers_count": followers,
            "following_count": following,
            "follower_following_ratio": round(follower_following_ratio, 3),
            "posts_count": posts,
            "avg_likes_per_post": round(avg_likes_per_post, 2),
            "avg_comments_per_post": round(avg_comments_per_post, 2),
            "posting_freq_per_week": round(posting_freq_per_week, 2),
            "account_age_days": account_age_days,
            "engagement_rate": round(min(engagement_rate, 1.0), 4),
            "night_activity_ratio": round(night_activity_ratio, 3),
            "is_fake": int(is_fake),
        })

    return pd.DataFrame(rows)


def generate_graph_features(df):
    """
    Build a synthetic follow-graph with realistic structure:
    - Genuine accounts form a scale-free network (preferential attachment)
    - Fake accounts cluster densely with each other (bot farms / follow-for-follow rings)
    """
    n = len(df)
    G = nx.barabasi_albert_graph(n=n, m=3, seed=RANDOM_SEED)

    id_map = {i: df.iloc[i]["account_id"] for i in range(n)}
    fake_nodes = [i for i in range(n) if df.iloc[i]["is_fake"] == 1]

    # Add dense fake-to-fake edges to simulate follow-for-follow bot rings
    for _ in range(len(fake_nodes) * 4):
        a, b = random.sample(fake_nodes, 2)
        G.add_edge(a, b)

    degree = dict(G.degree())
    clustering = nx.clustering(G)

    # betweenness centrality is expensive at scale; use an approximation via sampling
    betweenness = nx.betweenness_centrality(G, k=min(200, n), seed=RANDOM_SEED)

    graph_rows = []
    for i in range(n):
        graph_rows.append({
            "account_id": id_map[i],
            "graph_degree": degree.get(i, 0),
            "clustering_coefficient": round(clustering.get(i, 0.0), 4),
            "betweenness_centrality": round(betweenness.get(i, 0.0), 6),
        })

    return pd.DataFrame(graph_rows)


def main():
    print(f"Generating {N_ACCOUNTS} accounts (~{int(FAKE_RATIO*100)}% fake)...")
    profile_behavior_df = generate_profile_and_behavioral_features(N_ACCOUNTS, FAKE_RATIO)

    print("Building synthetic follow-graph and computing network features...")
    graph_df = generate_graph_features(profile_behavior_df)

    print("Merging profile + behavioral + graph layers...")
    full_df = profile_behavior_df.merge(graph_df, on="account_id")

    # reorder: label at the end
    label = full_df.pop("is_fake")
    full_df["is_fake"] = label

    out_path = "/home/claude/dataset/instagram_fake_account_dataset.csv"
    full_df.to_csv(out_path, index=False)

    print(f"\nSaved dataset: {out_path}")
    print(f"Shape: {full_df.shape}")
    print(f"Fake accounts: {full_df['is_fake'].sum()} ({full_df['is_fake'].mean()*100:.1f}%)")
    print("\nColumns:", list(full_df.columns))
    print("\nSample rows:")
    print(full_df.head(3).to_string())


if __name__ == "__main__":
    main()
