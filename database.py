from pathlib import Path
import sqlite3
import pandas as pd

DB = Path(__file__).parent / "data" / "fraud_alerts.db"

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            customer_id INTEGER,
            fraud_probability REAL,
            anomaly_score REAL,
            risk_score REAL,
            risk_level TEXT,
            reason TEXT,
            status TEXT DEFAULT 'OPEN',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

def save_alert(row):
    init_db()
    with sqlite3.connect(DB) as con:
        con.execute("""
        INSERT INTO alerts
        (transaction_id, customer_id, fraud_probability, anomaly_score,
         risk_score, risk_level, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (
            row["transaction_id"], int(row["customer_id"]),
            float(row["fraud_probability"]), float(row["anomaly_score"]),
            float(row["risk_score"]), row["risk_level"], row["reason"]
        ))

def get_alerts():
    init_db()
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query(
            "SELECT * FROM alerts ORDER BY id DESC", con
        )

def close_alert(alert_id):
    init_db()
    with sqlite3.connect(DB) as con:
        con.execute("UPDATE alerts SET status='REVIEWED' WHERE id=?", (int(alert_id),))
