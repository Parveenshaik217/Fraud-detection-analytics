import pandas as pd
import numpy as np

FEATURES = [
    "amount", "hour", "day_of_week", "is_weekend", "distance_km",
    "device_age_days", "account_age_days", "transactions_last_24h",
    "avg_amount_30d", "failed_attempts_24h", "is_new_device",
    "is_new_location", "merchant_risk"
]

CATEGORICAL = ["channel"]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["amount_to_avg"] = x["amount"] / (x["avg_amount_30d"] + 1)
    x["night_transaction"] = ((x["hour"] < 6) | (x["hour"] >= 23)).astype(int)
    x["velocity_risk"] = np.clip(x["transactions_last_24h"] / 10, 0, 1)
    x["failed_attempt_risk"] = np.clip(x["failed_attempts_24h"] / 5, 0, 1)
    x["channel_WEB"] = (x["channel"] == "WEB").astype(int)
    x["channel_MOBILE"] = (x["channel"] == "MOBILE").astype(int)
    x["channel_POS"] = (x["channel"] == "POS").astype(int)
    cols = FEATURES + [
        "amount_to_avg", "night_transaction", "velocity_risk",
        "failed_attempt_risk", "channel_WEB", "channel_MOBILE", "channel_POS"
    ]
    return x[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
