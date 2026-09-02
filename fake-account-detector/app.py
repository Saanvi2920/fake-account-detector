"""
Flask Web Application - Instagram Fake Account Detection & Auto-Reporting
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from auto_report import report_fake_account, batch_report, get_all_reports

app = Flask(__name__)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

ensemble_model = joblib.load(os.path.join(MODELS_DIR, "stack_ensemble.joblib"))
base_models = {
    name: joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
    for name in ["logistic_regression", "decision_tree", "random_forest", "xgboost", "ann"]
}

FEATURE_COLS = [
    "username_length", "username_digit_ratio", "has_profile_pic",
    "bio_length", "has_external_url", "is_private", "followers_count",
    "following_count", "follower_following_ratio", "posts_count",
    "avg_likes_per_post", "avg_comments_per_post", "posting_freq_per_week",
    "account_age_days", "engagement_rate", "night_activity_ratio",
    "graph_degree", "clustering_coefficient", "betweenness_centrality",
    "profile_completeness_score", "username_suspicion_score",
    "posts_per_day_alive", "engagement_deficit", "bot_activity_pattern",
    "aggressive_following_flag", "empty_identity_flag",
    "network_anomaly_score", "is_network_hub",
]


def engineer_features(row_dict):
    d = row_dict.copy()
    d["profile_completeness_score"] = (
        d["has_profile_pic"] + d["has_external_url"] +
        (1 if d["bio_length"] > 20 else 0)
    ) / 3
    d["username_suspicion_score"] = (
        (0.5 if d["username_length"] > 12 else 0) +
        (0.5 if d["username_digit_ratio"] > 0.3 else 0)
    )
    d["posts_per_day_alive"] = d["posts_count"] / max(d["account_age_days"], 1)
    expected_eng = np.log1p(d["followers_count"]) * 0.05
    d["engagement_deficit"] = expected_eng - d["engagement_rate"]
    d["bot_activity_pattern"] = d["night_activity_ratio"] * (1 / (d["posting_freq_per_week"] + 0.1))
    d["aggressive_following_flag"] = 1 if d["follower_following_ratio"] < 0.3 else 0
    d["empty_identity_flag"] = 1 if (d["bio_length"] < 10 and d["has_profile_pic"] == 0) else 0
    d["network_anomaly_score"] = d["graph_degree"] * (1 - d["clustering_coefficient"])
    return d


def predict_account(features_dict):
    engineered = engineer_features(features_dict)
    X = pd.DataFrame([{col: engineered.get(col, 0) for col in FEATURE_COLS}])
    base_probas = np.array([
        base_models[name].predict_proba(X)[:, 1][0]
        for name in base_models
    ]).reshape(1, -1)
    proba = ensemble_model.predict_proba(base_probas)[0][1]
    label = "FAKE" if proba >= 0.5 else "GENUINE"
    return label, proba


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Instagram Fake Account Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f0f23; color: #e0e0e0; }
        .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
        header { text-align: center; padding: 30px 0; border-bottom: 2px solid #1a1a3e; margin-bottom: 30px; }
        header h1 { font-size: 2.2em; color: #00d4ff; }
        header p { color: #888; margin-top: 8px; }
        .nav { display: flex; justify-content: center; gap: 12px; margin-bottom: 30px; flex-wrap: wrap; }
        .nav a { padding: 10px 24px; background: #1a1a3e; color: #00d4ff; border-radius: 8px;
                 text-decoration: none; font-weight: 600; transition: all 0.3s; border: 1px solid #2a2a5e; }
        .nav a:hover, .nav a.active { background: #00d4ff; color: #0f0f23; }
        .card { background: #1a1a2e; border-radius: 12px; padding: 30px; margin-bottom: 24px; border: 1px solid #2a2a5e; }
        .card h2 { color: #00d4ff; margin-bottom: 20px; font-size: 1.4em; }
        label { display: block; margin-bottom: 5px; color: #aaa; font-size: 0.9em; }
        input, select { width: 100%; padding: 10px 14px; margin-bottom: 16px; border-radius: 8px;
                        border: 1px solid #333; background: #0f0f23; color: #e0e0e0; font-size: 1em; }
        input:focus { outline: none; border-color: #00d4ff; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
        button { padding: 12px 32px; background: #00d4ff; color: #0f0f23; border: none; border-radius: 8px;
                 font-size: 1.1em; font-weight: 700; cursor: pointer; transition: all 0.3s; }
        button:hover { background: #00b8d9; transform: translateY(-1px); }
        .result { margin-top: 20px; padding: 20px; border-radius: 10px; }
        .result.fake { background: #3d1111; border: 2px solid #ff4444; }
        .result.genuine { background: #113d11; border: 2px solid #44ff44; }
        .result h3 { font-size: 1.3em; margin-bottom: 10px; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 700; }
        .badge-critical { background: #ff2222; color: #fff; }
        .badge-high { background: #ff6600; color: #fff; }
        .badge-medium { background: #ffaa00; color: #000; }
        .badge-low { background: #44bb44; color: #000; }
        .badge-fake { background: #ff4444; color: #fff; }
        .badge-genuine { background: #44ff44; color: #000; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #2a2a5e; }
        th { color: #00d4ff; font-weight: 600; background: #0f0f23; }
        tr:hover { background: #16213e; }
        .stats-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .stat-card { flex: 1; min-width: 160px; background: #16213e; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #2a2a5e; }
        .stat-card .number { font-size: 2em; font-weight: 700; color: #00d4ff; }
        .stat-card .label { color: #888; margin-top: 4px; }
        .reasons { list-style: none; padding: 0; }
        .reasons li { padding: 6px 0; color: #ff8888; }
        .report-card { background: #16213e; border-radius: 10px; padding: 16px; margin-bottom: 12px; border-left: 4px solid #ff4444; }
        .report-card.critical { border-left-color: #ff2222; }
        .report-card.high { border-left-color: #ff6600; }
        .report-card.medium { border-left-color: #ffaa00; }
        .report-card.low { border-left-color: #44bb44; }
        .flex-between { display: flex; justify-content: space-between; align-items: center; }
        .empty { text-align: center; padding: 40px; color: #666; }
        @media (max-width: 768px) { .grid-3 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Instagram Fake Account Detector</h1>
            <p>Auto-detection and automatic reporting of fake/spam accounts</p>
        </header>
        <div class="nav">
            <a href="/" class="{{ 'active' if active == 'predict' }}">Predict</a>
            <a href="/bulk" class="{{ 'active' if active == 'bulk' }}">Bulk Detection</a>
            <a href="/reports" class="{{ 'active' if active == 'reports' }}">Reports</a>
            <a href="/demo" class="{{ 'active' if active == 'demo' }}">Run Demo</a>
        </div>
        {{ content|safe }}
    </div>
</body>
</html>"""

PREDICT_PAGE = """
<div class="card">
    <h2>Single Account Prediction</h2>
    <form method="POST" action="/predict">
        <div class="grid">
            <div><label>Account ID</label><input type="text" name="account_id" placeholder="acc_9999" required></div>
            <div><label>Username</label><input type="text" name="username" placeholder="user_name"></div>
        </div>
        <div class="grid-3">
            <div><label>Username Length</label><input type="number" name="username_length" value="10" step="1"></div>
            <div><label>Username Digit Ratio</label><input type="number" name="username_digit_ratio" value="0.1" step="0.01"></div>
            <div><label>Has Profile Pic (0/1)</label><input type="number" name="has_profile_pic" value="1" min="0" max="1"></div>
        </div>
        <div class="grid-3">
            <div><label>Bio Length</label><input type="number" name="bio_length" value="50" step="1"></div>
            <div><label>Has External URL (0/1)</label><input type="number" name="has_external_url" value="0" min="0" max="1"></div>
            <div><label>Is Private (0/1)</label><input type="number" name="is_private" value="0" min="0" max="1"></div>
        </div>
        <div class="grid-3">
            <div><label>Followers Count</label><input type="number" name="followers_count" value="200"></div>
            <div><label>Following Count</label><input type="number" name="following_count" value="300"></div>
            <div><label>Follower/Following Ratio</label><input type="number" name="follower_following_ratio" value="0.67" step="0.01"></div>
        </div>
        <div class="grid-3">
            <div><label>Posts Count</label><input type="number" name="posts_count" value="30"></div>
            <div><label>Avg Likes/Post</label><input type="number" name="avg_likes_per_post" value="15" step="0.1"></div>
            <div><label>Avg Comments/Post</label><input type="number" name="avg_comments_per_post" value="2" step="0.1"></div>
        </div>
        <div class="grid-3">
            <div><label>Posting Freq/Week</label><input type="number" name="posting_freq_per_week" value="2" step="0.1"></div>
            <div><label>Account Age (days)</label><input type="number" name="account_age_days" value="365"></div>
            <div><label>Engagement Rate</label><input type="number" name="engagement_rate" value="0.05" step="0.001"></div>
        </div>
        <div class="grid-3">
            <div><label>Night Activity Ratio</label><input type="number" name="night_activity_ratio" value="0.2" step="0.01"></div>
            <div><label>Graph Degree</label><input type="number" name="graph_degree" value="50"></div>
            <div><label>Clustering Coefficient</label><input type="number" name="clustering_coefficient" value="0.01" step="0.001"></div>
        </div>
        <div class="grid-3">
            <div><label>Betweenness Centrality</label><input type="number" name="betweenness_centrality" value="0.02" step="0.001"></div>
            <div></div><div></div>
        </div>
        <button type="submit">Detect Account</button>
    </form>
</div>
{% if result %}
<div class="result {{ 'fake' if result.label == 'FAKE' else 'genuine' }}">
    <h3>
        {% if result.label == 'FAKE' %}FAKE ACCOUNT DETECTED - AUTO-REPORT GENERATED{% else %}GENUINE ACCOUNT{% endif %}
        <span class="badge {{ 'badge-fake' if result.label == 'FAKE' else 'badge-genuine' }}">{{ result.label }}</span>
    </h3>
    <p style="margin-top:6px;color:#aaa;">Confidence: {{ "%.1f"|format(result.confidence * 100) }}%</p>
    {% if result.label == 'FAKE' %}
    <div style="margin-top:12px;">
        <strong style="color:#ff4444;">Report ID:</strong> {{ result.report.report_id }}<br>
        <strong>Risk Level:</strong> <span class="badge badge-{{ result.report.risk_level|lower }}">{{ result.report.risk_level }}</span><br>
        <strong>Action:</strong> {{ result.report.recommended_action }}
    </div>
    <div style="margin-top:12px;"><strong>Reasons:</strong><ul class="reasons">{% for r in result.report.suspicion_reasons %}<li>{{ r }}</li>{% endfor %}</ul></div>
    {% endif %}
</div>
{% endif %}"""

BULK_PAGE = """
<div class="card">
    <h2>Bulk Account Detection</h2>
    <p style="color:#888;margin-bottom:16px;">Scans all test accounts. Detected fakes are automatically reported.</p>
    <form method="POST" action="/bulk"><button type="submit">Scan Dataset & Auto-Report All Fakes</button></form>
</div>
{% if bulk_result %}
<div class="card">
    <h2>Scan Results</h2>
    <div class="stats-row">
        <div class="stat-card"><div class="number">{{ bulk_result.total }}</div><div class="label">Scanned</div></div>
        <div class="stat-card"><div class="number" style="color:#ff4444;">{{ bulk_result.fakes }}</div><div class="label">Fake Detected</div></div>
        <div class="stat-card"><div class="number" style="color:#44ff44;">{{ bulk_result.genuine }}</div><div class="label">Genuine</div></div>
        <div class="stat-card"><div class="number" style="color:#ffaa00;">{{ bulk_result.reports_generated }}</div><div class="label">Reports Generated</div></div>
    </div>
    {% if bulk_result.sample_reports %}
    <h3 style="margin-top:16px;">Top 10 Highest Risk Auto-Reports</h3>
    {% for r in bulk_result.sample_reports %}
    <div class="report-card {{ r.risk_level|lower }}">
        <div class="flex-between">
            <div><strong>{{ r.account_id }}</strong> <span class="badge badge-{{ r.risk_level|lower }}">{{ r.risk_level }}</span></div>
            <span style="color:#888;">{{ r.report_id }}</span>
        </div>
        <p style="margin-top:8px;color:#aaa;font-size:0.9em;">Confidence: {{ "%.1f"|format(r.confidence * 100) }}% | Reasons: {{ r.reasons|length }}</p>
    </div>
    {% endfor %}{% endif %}
</div>{% endif %}"""

REPORTS_PAGE = """
<div class="card">
    <h2>All Auto-Generated Reports</h2>
    <div class="stats-row">
        <div class="stat-card"><div class="number">{{ stats.total }}</div><div class="label">Total</div></div>
        <div class="stat-card"><div class="number" style="color:#ff2222;">{{ stats.critical }}</div><div class="label">Critical</div></div>
        <div class="stat-card"><div class="number" style="color:#ff6600;">{{ stats.high }}</div><div class="label">High</div></div>
        <div class="stat-card"><div class="number" style="color:#ffaa00;">{{ stats.medium }}</div><div class="label">Medium</div></div>
        <div class="stat-card"><div class="number" style="color:#44bb44;">{{ stats.low }}</div><div class="label">Low</div></div>
    </div>
    {% if reports %}
    <table>
        <thead><tr><th>Report ID</th><th>Account</th><th>Risk</th><th>Confidence</th><th>Status</th><th>Time</th></tr></thead>
        <tbody>{% for r in reports %}<tr>
            <td>{{ r.report_id }}</td><td>{{ r.account_id }}</td>
            <td><span class="badge badge-{{ r.risk_level|lower }}">{{ r.risk_level }}</span></td>
            <td>{{ "%.1f"|format(r.confidence_score * 100) }}%</td><td>{{ r.status }}</td>
            <td>{{ r.detection_timestamp[:19] }}</td>
        </tr>{% endfor %}</tbody>
    </table>
    {% else %}<div class="empty">No reports yet. Run a scan first.</div>{% endif %}
</div>"""

DEMO_PAGE = """
<div class="card">
    <h2>Run Full Demo</h2>
    <p style="color:#888;margin-bottom:16px;">Runs the full pipeline on the test dataset with auto-reporting.</p>
    <form method="POST" action="/demo"><button type="submit">Run Full Demo</button></form>
</div>
{% if demo_result %}
<div class="card">
    <h2>Demo Results</h2>
    <div class="stats-row">
        <div class="stat-card"><div class="number">{{ demo_result.total }}</div><div class="label">Test Accounts</div></div>
        <div class="stat-card"><div class="number" style="color:#ff4444;">{{ demo_result.fakes_detected }}</div><div class="label">Fakes Detected</div></div>
        <div class="stat-card"><div class="number" style="color:#ffaa00;">{{ demo_result.reports_generated }}</div><div class="label">Reports Auto-Generated</div></div>
    </div>
    <h3 style="margin-top:20px;">Model Performance</h3>
    <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
            <tr><td>Accuracy</td><td>{{ "%.4f"|format(demo_result.metrics.accuracy) }}</td></tr>
            <tr><td>Precision</td><td>{{ "%.4f"|format(demo_result.metrics.precision) }}</td></tr>
            <tr><td>Recall</td><td>{{ "%.4f"|format(demo_result.metrics.recall) }}</td></tr>
            <tr><td>F1 Score</td><td>{{ "%.4f"|format(demo_result.metrics.f1) }}</td></tr>
            <tr><td>ROC AUC</td><td>{{ "%.4f"|format(demo_result.metrics.roc_auc) }}</td></tr>
        </tbody>
    </table>
    {% if demo_result.sample_reports %}
    <h3 style="margin-top:20px;">Sample Auto-Reports</h3>
    {% for r in demo_result.sample_reports %}
    <div class="report-card {{ r.risk_level|lower }}">
        <div class="flex-between">
            <div><strong>{{ r.account_id }}</strong> <span class="badge badge-{{ r.risk_level|lower }}">{{ r.risk_level }}</span></div>
            <span style="color:#888;">{{ r.report_id }}</span>
        </div>
        <p style="margin-top:6px;color:#aaa;font-size:0.9em;">Confidence: {{ "%.1f"|format(r.confidence * 100) }}%</p>
        <ul class="reasons" style="margin-top:6px;">{% for reason in r.reasons[:3] %}<li style="font-size:0.85em;">{{ reason }}</li>{% endfor %}</ul>
    </div>
    {% endfor %}{% endif %}
</div>{% endif %}"""


@app.route("/")
def index():
    content = render_template_string(PREDICT_PAGE, result=None)
    return render_template_string(HTML_TEMPLATE, content=content, active="predict")


@app.route("/predict", methods=["POST"])
def predict():
    features = {}
    for field in request.form:
        val = request.form[field]
        if field in ("account_id", "username"):
            features[field] = val
        else:
            try:
                features[field] = float(val)
            except ValueError:
                features[field] = 0

    account_id = features.pop("account_id", "unknown")
    username = features.pop("username", account_id)
    label, confidence = predict_account(features)
    result = {"label": label, "confidence": confidence, "report": None}

    if label == "FAKE":
        report = report_fake_account(
            account_id=account_id, confidence_score=confidence,
            account_features=features, username=username,
        )
        result["report"] = report

    content = render_template_string(PREDICT_PAGE, result=result)
    return render_template_string(HTML_TEMPLATE, content=content, active="predict")


@app.route("/bulk", methods=["GET", "POST"])
def bulk():
    if request.method == "GET":
        content = render_template_string(BULK_PAGE, bulk_result=None)
        return render_template_string(HTML_TEMPLATE, content=content, active="bulk")

    test = pd.read_csv(os.path.join(DATA_DIR, "ml_ready_test.csv"))
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "instagram_fake_account_dataset.csv"))

    feature_cols = [c for c in test.columns if c != "is_fake"]
    X_test = test[feature_cols]

    base_probas = np.column_stack([
        base_models[name].predict_proba(X_test)[:, 1] for name in base_models
    ])
    ensemble_probas = ensemble_model.predict_proba(base_probas)[:, 1]
    predictions = (ensemble_probas >= 0.5).astype(int)

    total = len(test)
    fakes = int(predictions.sum())
    fake_indices = np.where(predictions == 1)[0]
    reports_for_display = []

    for idx in fake_indices[:50]:
        row = test_raw.iloc[idx] if idx < len(test_raw) else None
        acc_id = row["account_id"] if row is not None else f"acc_{idx}"
        feat = {col: float(row[col]) for col in test_raw.columns if col not in ("account_id", "is_fake") and row is not None}
        report = report_fake_account(
            account_id=acc_id, confidence_score=float(ensemble_probas[idx]),
            account_features=feat, username=f"user_{idx}",
        )
        reports_for_display.append({
            "account_id": acc_id, "report_id": report["report_id"],
            "confidence": ensemble_probas[idx], "risk_level": report["risk_level"],
            "reasons": report["suspicion_reasons"],
        })

    reports_for_display.sort(key=lambda x: x["confidence"], reverse=True)
    content = render_template_string(BULK_PAGE, bulk_result={
        "total": total, "fakes": fakes, "genuine": total - fakes,
        "reports_generated": fakes, "sample_reports": reports_for_display[:10],
    })
    return render_template_string(HTML_TEMPLATE, content=content, active="bulk")


@app.route("/reports")
def reports():
    all_reports = get_all_reports()
    stats = {"total": len(all_reports), "critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in all_reports:
        stats[r.get("risk_level", "LOW").lower()] = stats.get(r.get("risk_level", "LOW").lower(), 0) + 1
    content = render_template_string(REPORTS_PAGE, reports=all_reports, stats=stats)
    return render_template_string(HTML_TEMPLATE, content=content, active="reports")


@app.route("/demo", methods=["GET", "POST"])
def demo():
    if request.method == "GET":
        content = render_template_string(DEMO_PAGE, demo_result=None)
        return render_template_string(HTML_TEMPLATE, content=content, active="demo")

    test = pd.read_csv(os.path.join(DATA_DIR, "ml_ready_test.csv"))
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "instagram_fake_account_dataset.csv"))

    feature_cols = [c for c in test.columns if c != "is_fake"]
    X_test, y_test = test[feature_cols], test["is_fake"]

    base_probas = np.column_stack([
        base_models[name].predict_proba(X_test)[:, 1] for name in base_models
    ])
    y_proba = ensemble_model.predict_proba(base_probas)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    fake_indices = np.where(y_pred == 1)[0]
    np.random.seed(42)
    sample_indices = np.random.choice(fake_indices, size=min(10, len(fake_indices)), replace=False)

    sample_reports = []
    for idx in sample_indices:
        row = test_raw.iloc[idx] if idx < len(test_raw) else None
        acc_id = row["account_id"] if row is not None else f"acc_{idx}"
        feat = {col: float(row[col]) for col in test_raw.columns if col not in ("account_id", "is_fake") and row is not None}
        report = report_fake_account(
            account_id=acc_id, confidence_score=float(y_proba[idx]),
            account_features=feat, username=f"user_{idx}",
        )
        sample_reports.append({
            "account_id": acc_id, "report_id": report["report_id"],
            "confidence": y_proba[idx], "risk_level": report["risk_level"],
            "reasons": report["suspicion_reasons"],
        })
    sample_reports.sort(key=lambda x: x["confidence"], reverse=True)

    content = render_template_string(DEMO_PAGE, demo_result={
        "total": len(test), "fakes_detected": int(y_pred.sum()),
        "reports_generated": int(y_pred.sum()), "metrics": metrics,
        "sample_reports": sample_reports,
    })
    return render_template_string(HTML_TEMPLATE, content=content, active="demo")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
