"""
Step 7: Auto-Report System for Fake Accounts
----------------------------------------------
When the model detects a fake account, this module automatically:
  1. Logs the detection event with full details
  2. Generates a structured report (JSON + human-readable text)
  3. Flags the account for review
  4. Maintains a running report database (SQLite)
  5. Can trigger email/webhook notifications (simulated)
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "reports", "fake_account_reports.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def init_database():
    """Initialize SQLite database for report storage."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            username TEXT,
            detection_timestamp TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            top_suspicion_reasons TEXT NOT NULL,
            model_used TEXT NOT NULL,
            status TEXT DEFAULT 'pending_review',
            action_taken TEXT DEFAULT 'none'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS report_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            notes TEXT,
            FOREIGN KEY (report_id) REFERENCES reports(report_id)
        )
    """)
    conn.commit()
    return conn


def compute_risk_level(confidence):
    """Classify risk based on model confidence."""
    if confidence >= 0.9:
        return "CRITICAL"
    elif confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def identify_suspicion_reasons(account_features):
    """Identify the top reasons why this account is flagged as fake."""
    reasons = []

    if account_features.get("username_digit_ratio", 0) > 0.3:
        reasons.append("High digit ratio in username (bot-generated name)")

    if account_features.get("has_profile_pic", 1) == 0:
        reasons.append("No profile picture uploaded")

    if account_features.get("bio_length", 100) < 10:
        reasons.append("Empty or minimal bio")

    followers = account_features.get("followers_count", 100)
    following = account_features.get("following_count", 100)
    if following > 0 and followers / max(following, 1) < 0.1:
        reasons.append(f"Severe follower/following imbalance ({followers}/{following})")

    if account_features.get("night_activity_ratio", 0) > 0.5:
        reasons.append("Majority of activity during night hours (bot pattern)")

    if account_features.get("posts_count", 10) < 3:
        reasons.append(f"Very few posts ({account_features.get('posts_count', 0)})")

    if account_features.get("engagement_rate", 0.1) < 0.01:
        reasons.append("Near-zero engagement rate")

    if account_features.get("account_age_days", 365) < 30:
        reasons.append(f"Very new account ({account_features.get('account_age_days', 0)} days old)")

    if account_features.get("aggressive_following_flag", 0) == 1:
        reasons.append("Aggressive following pattern (follow-for-follow)")

    if account_features.get("empty_identity_flag", 0) == 1:
        reasons.append("Empty identity: no profile pic + short bio")

    if account_features.get("bot_activity_pattern", 0) > 2:
        reasons.append("Abnormal bot-like activity pattern")

    if not reasons:
        reasons.append("Multiple weak signals combined by ensemble model")

    return reasons


def generate_report_id(account_id):
    """Generate a unique report ID."""
    timestamp = datetime.now().isoformat()
    raw = f"{account_id}_{timestamp}"
    return "RPT-" + hashlib.md5(raw.encode()).hexdigest()[:12].upper()


def report_fake_account(account_id, confidence_score, account_features,
                        model_used="stack_ensemble", username=None):
    conn = init_database()
    c = conn.cursor()

    report_id = generate_report_id(account_id)
    risk_level = compute_risk_level(confidence_score)
    reasons = identify_suspicion_reasons(account_features)
    timestamp = datetime.now().isoformat()

    c.execute("""
        INSERT INTO reports (report_id, account_id, username, detection_timestamp,
                             confidence_score, risk_level, top_suspicion_reasons,
                             model_used, status, action_taken)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id, account_id, username, timestamp,
        confidence_score, risk_level, json.dumps(reasons),
        model_used, "pending_review", "auto_flagged"
    ))

    c.execute("""
        INSERT INTO report_actions (report_id, action, timestamp, notes)
        VALUES (?, ?, ?, ?)
    """, (
        report_id, "auto_report_generated", timestamp,
        f"Account auto-flagged with {risk_level} risk ({confidence_score:.1%} confidence)"
    ))

    conn.commit()

    report = {
        "report_id": report_id,
        "account_id": account_id,
        "username": username or account_id,
        "timestamp": timestamp,
        "confidence_score": round(confidence_score, 4),
        "risk_level": risk_level,
        "suspicion_reasons": reasons,
        "model_used": model_used,
        "status": "pending_review",
        "action_taken": "auto_flagged",
        "recommended_action": _get_recommended_action(risk_level),
    }

    report_path = os.path.join(REPORTS_DIR, f"{report_id}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    conn.close()
    return report


def _get_recommended_action(risk_level):
    actions = {
        "CRITICAL": "Immediate suspension and manual investigation required",
        "HIGH": "Flag for priority review, restrict account activity",
        "MEDIUM": "Queue for standard review within 24 hours",
        "LOW": "Monitor and log, no immediate action needed",
    }
    return actions.get(risk_level, "Log for review")


def batch_report(fake_accounts_list):
    reports = []
    for account in fake_accounts_list:
        report = report_fake_account(
            account_id=account["account_id"],
            confidence_score=account["confidence_score"],
            account_features=account.get("account_features", {}),
            username=account.get("username"),
        )
        reports.append(report)

    summary = generate_batch_summary(reports)
    return reports, summary


def generate_batch_summary(reports):
    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in reports:
        risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1

    summary = {
        "batch_timestamp": datetime.now().isoformat(),
        "total_reports": len(reports),
        "risk_distribution": risk_counts,
        "accounts_requiring_immediate_action": risk_counts["CRITICAL"] + risk_counts["HIGH"],
        "reports": [r["report_id"] for r in reports],
    }

    summary_path = os.path.join(REPORTS_DIR, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def get_report_status(report_id):
    conn = init_database()
    c = conn.cursor()
    c.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {
            "report_id": row[0], "account_id": row[1], "username": row[2],
            "detection_timestamp": row[3], "confidence_score": row[4],
            "risk_level": row[5], "top_suspicion_reasons": json.loads(row[6]),
            "model_used": row[7], "status": row[8], "action_taken": row[9],
        }
    return None


def get_all_reports(status_filter=None):
    conn = init_database()
    c = conn.cursor()
    if status_filter:
        c.execute("SELECT * FROM reports WHERE status = ? ORDER BY detection_timestamp DESC",
                  (status_filter,))
    else:
        c.execute("SELECT * FROM reports ORDER BY detection_timestamp DESC")
    rows = c.fetchall()
    conn.close()

    return [
        {
            "report_id": r[0], "account_id": r[1], "username": r[2],
            "detection_timestamp": r[3], "confidence_score": r[4],
            "risk_level": r[5], "status": r[8],
        }
        for r in rows
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("AUTO-REPORT SYSTEM DEMO")
    print("=" * 60)

    test_accounts = [
        {
            "account_id": "acc_3",
            "username": "bot_user_9283",
            "confidence_score": 0.94,
            "account_features": {
                "username_digit_ratio": 0.796, "has_profile_pic": 0,
                "bio_length": 11, "followers_count": 131, "following_count": 251,
                "posts_count": 0, "night_activity_ratio": 0.624,
                "engagement_rate": 0.0048, "account_age_days": 142,
                "aggressive_following_flag": 1, "empty_identity_flag": 1,
            },
        },
        {
            "account_id": "acc_4",
            "username": "sp4m_acc_5521",
            "confidence_score": 0.89,
            "account_features": {
                "username_digit_ratio": 0.395, "has_profile_pic": 0,
                "bio_length": 5, "followers_count": 70, "following_count": 889,
                "posts_count": 3, "night_activity_ratio": 0.759,
                "engagement_rate": 0.0012, "account_age_days": 26,
                "aggressive_following_flag": 1, "empty_identity_flag": 1,
            },
        },
        {
            "account_id": "acc_5",
            "username": "fake_promo_7744",
            "confidence_score": 0.97,
            "account_features": {
                "username_digit_ratio": 0.742, "has_profile_pic": 1,
                "bio_length": 6, "followers_count": 5, "following_count": 663,
                "posts_count": 5, "night_activity_ratio": 0.698,
                "engagement_rate": 0.2735, "account_age_days": 110,
                "aggressive_following_flag": 1, "empty_identity_flag": 0,
            },
        },
    ]

    reports, summary = batch_report(test_accounts)

    for report in reports:
        print(f"\n{'-'*50}")
        print(f"REPORT: {report['report_id']}")
        print(f"Account: {report['username']} ({report['account_id']})")
        print(f"Risk Level: {report['risk_level']}")
        print(f"Confidence: {report['confidence_score']:.1%}")
        print(f"Reasons:")
        for reason in report["suspicion_reasons"]:
            print(f"  - {reason}")
        print(f"Recommended Action: {report['recommended_action']}")

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total Reports: {summary['total_reports']}")
    print(f"Risk Distribution: {summary['risk_distribution']}")
    print(f"Require Immediate Action: {summary['accounts_requiring_immediate_action']}")
