import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 30000

customer = RNG.integers(10000, 15000, N)
amount = np.round(np.exp(RNG.normal(np.log(1800), 1.0, N)), 2)
hour = RNG.integers(0, 24, N)
dow = RNG.integers(0, 7, N)
distance = np.round(np.abs(RNG.normal(18, 35, N)), 2)
device_age = RNG.integers(1, 1500, N)
account_age = RNG.integers(30, 2500, N)
tx24 = RNG.poisson(3, N)
avg30 = np.round(np.exp(RNG.normal(np.log(1500), .65, N)), 2)
failed = RNG.poisson(.35, N)
new_device = RNG.binomial(1, .12, N)
new_location = RNG.binomial(1, .10, N)
merchant_risk = np.round(RNG.beta(2, 5, N), 3)
channel = RNG.choice(["POS", "WEB", "MOBILE"], N, p=[.35, .30, .35])

# Synthetic but intentionally non-trivial fraud signal.
logit = (
    -5.0
    + 1.15 * new_device
    + 1.10 * new_location
    + 0.018 * distance
    + 0.00055 * amount
    + 0.55 * (hour < 5)
    + 0.38 * (tx24 > 8)
    + 0.70 * (failed >= 3)
    + 1.15 * merchant_risk
    + 0.45 * (amount > 5000)
    + 0.30 * (channel == "WEB")
)
p = 1 / (1 + np.exp(-logit))
fraud = RNG.binomial(1, np.clip(p, 0, .95))

df = pd.DataFrame({
    "transaction_id": [f"TX{i:07d}" for i in range(1, N + 1)],
    "customer_id": customer,
    "amount": amount,
    "hour": hour,
    "day_of_week": dow,
    "is_weekend": (dow >= 5).astype(int),
    "distance_km": distance,
    "device_age_days": device_age,
    "account_age_days": account_age,
    "transactions_last_24h": tx24,
    "avg_amount_30d": avg30,
    "failed_attempts_24h": failed,
    "is_new_device": new_device,
    "is_new_location": new_location,
    "merchant_risk": merchant_risk,
    "channel": channel,
    "fraud": fraud
})

# Keep the problem realistic enough for imbalanced-learning metrics.
df.to_csv(Path(__file__).parent / "data" / "transactions.csv", index=False)
print(f"Created {len(df):,} transactions. Fraud rate: {df.fraud.mean():.2%}")
