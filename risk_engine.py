import numpy as np
import pandas as pd

def anomaly_probability(iso_model, X):
    # Isolation Forest: -1 anomaly, +1 normal.
    raw = -iso_model.decision_function(X)
    # Stable min-max normalization for the scored batch.
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return np.full(len(X), .5)
    return np.clip((raw - lo) / (hi - lo), 0, 1)

def calculate_risk(fraud_prob, anomaly_prob, row):
    velocity = min(float(row["transactions_last_24h"]) / 10, 1)
    novelty = (float(row["is_new_device"]) + float(row["is_new_location"])) / 2
    merchant = float(row["merchant_risk"])
    amount_ratio = min(float(row["amount"]) / (float(row["avg_amount_30d"]) + 1), 5) / 5

    score = (
        .55 * fraud_prob +
        .20 * anomaly_prob +
        .10 * velocity +
        .08 * novelty +
        .04 * merchant +
        .03 * amount_ratio
    ) * 100

    level = "LOW" if score < 30 else "MEDIUM" if score < 60 else "HIGH" if score < 80 else "CRITICAL"
    return round(float(score), 2), level

def explain(row, fraud_prob):
    reasons = []
    if row["is_new_device"]: reasons.append("new device")
    if row["is_new_location"]: reasons.append("new location")
    if row["transactions_last_24h"] >= 8: reasons.append("high transaction velocity")
    if row["failed_attempts_24h"] >= 3: reasons.append("multiple failed attempts")
    if row["merchant_risk"] >= .65: reasons.append("high-risk merchant")
    if row["amount"] > 5000: reasons.append("unusually high amount")
    if row["hour"] < 6: reasons.append("unusual transaction time")
    return ", ".join(reasons[:5]) if reasons else "no major rule-based warning"
