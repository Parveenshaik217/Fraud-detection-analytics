# Real-Time Fraud Detection & Financial Risk Analytics System

A portfolio-ready Data Science project that combines supervised fraud classification,
unsupervised anomaly detection, real-time transaction simulation, risk scoring,
customer risk profiling, alert logging, and model monitoring.

## 3 extra features added

1. **Customer Risk Profiling** – aggregates recent transactions to calculate a customer-level risk score.
2. **Real-Time Alert & Case Management** – automatically creates HIGH/CRITICAL alerts and lets an analyst mark them as reviewed.
3. **Model/Data Drift Monitoring** – compares incoming transaction features with the training distribution and reports PSI-style drift.

## Main pipeline

Transaction -> validation -> feature engineering -> XGBoost fraud model
-> Isolation Forest anomaly model -> risk score -> alert engine -> dashboard

## Important note

The included dataset is synthetic so the project runs immediately without needing a private banking dataset.
For a real portfolio, you can later replace `data/transactions.csv` with a public fraud dataset after adapting
the feature mapping.

## Run on Windows

```text
1. Install Python 3.10+.
2. Open Command Prompt in this project folder.
3. Create an environment:
   python -m venv .venv
4. Activate it:
   .venv\Scripts\activate
5. Install dependencies:
   pip install -r requirements.txt
6. Train models:
   python train.py
7. Start the dashboard:
   streamlit run app.py
```

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the local Streamlit address shown in the terminal, usually:
`http://localhost:8501`

## Dashboard pages

- Overview
- Live Transaction Scoring
- Fraud Analytics
- Customer Risk
- Alerts
- Drift Monitoring

## CSV columns

The generated dataset contains:
- transaction_id
- customer_id
- amount
- hour
- day_of_week
- is_weekend
- distance_km
- device_age_days
- account_age_days
- transactions_last_24h
- avg_amount_30d
- failed_attempts_24h
- is_new_device
- is_new_location
- merchant_risk
- channel
- fraud

Replace the generated data only after keeping these feature names or updating `features.py`.
