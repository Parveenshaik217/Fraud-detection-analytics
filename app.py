from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from features import prepare_features
from risk_engine import anomaly_probability, calculate_risk, explain
from database import init_db, save_alert, get_alerts, close_alert
from drift import drift_table

BASE = Path(__file__).parent
DATA = BASE / "data" / "transactions.csv"
MODEL_DIR = BASE / "models"

st.set_page_config(page_title="Fraud Risk Analytics", page_icon="🛡️", layout="wide")
init_db()

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_models():
    return (
        joblib.load(MODEL_DIR / "fraud_model.joblib"),
        joblib.load(MODEL_DIR / "anomaly_model.joblib")
    )

@st.cache_data
def load_metrics():
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)

if not (MODEL_DIR / "fraud_model.joblib").exists():
    st.error("Models are not trained yet. Run: python train.py")
    st.stop()

df = load_data()
model, iso = load_models()
metrics = load_metrics()

st.sidebar.title("🛡️ FraudGuard")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Live Transaction", "Fraud Analytics",
     "Customer Risk", "Alerts", "Drift Monitoring"]
)

st.sidebar.caption("Portfolio Data Science Project")
st.sidebar.metric("Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud rate", f"{df.fraud.mean():.2%}")

if page == "Overview":
    st.title("Real-Time Fraud Detection & Financial Risk Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c3.metric("Precision", f"{metrics['precision']:.3f}")
    c4.metric("Recall", f"{metrics['recall']:.3f}")

    st.subheader("Fraud distribution")
    chart = px.histogram(df, x="amount", color="fraud", nbins=50,
                         title="Transaction Amount Distribution")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Risk signals")
    st.write(
        "The system combines supervised XGBoost fraud probability, "
        "Isolation Forest anomaly probability, transaction velocity, "
        "device/location novelty, merchant risk and amount deviation."
    )

elif page == "Live Transaction":
    st.title("⚡ Live Transaction Scoring")
    st.write("Enter a transaction to simulate a real-time fraud decision.")

    with st.form("transaction"):
        c1, c2, c3 = st.columns(3)
        txid = c1.text_input("Transaction ID", "LIVE-0001")
        customer_id = c2.number_input("Customer ID", 10000, 99999, 12001)
        amount = c3.number_input("Amount (₹)", 1.0, 500000.0, 85000.0)

        c1, c2, c3 = st.columns(3)
        hour = c1.slider("Hour", 0, 23, 2)
        dow = c2.slider("Day of week (0=Mon)", 0, 6, 2)
        distance = c3.number_input("Distance from usual location (km)", 0.0, 5000.0, 240.0)

        c1, c2, c3 = st.columns(3)
        device_age = c1.number_input("Device age (days)", 1, 5000, 2)
        account_age = c2.number_input("Account age (days)", 30, 5000, 500)
        tx24 = c3.number_input("Transactions in last 24h", 0, 100, 12)

        c1, c2, c3 = st.columns(3)
        avg30 = c1.number_input("Customer average amount (₹)", 1.0, 500000.0, 1500.0)
        failed = c2.number_input("Failed attempts in 24h", 0, 30, 4)
        merchant = c3.slider("Merchant risk", 0.0, 1.0, .75)

        c1, c2, c3 = st.columns(3)
        new_device = c1.checkbox("New device")
        new_location = c2.checkbox("New location")
        channel = c3.selectbox("Channel", ["POS", "WEB", "MOBILE"])

        submitted = st.form_submit_button("Score Transaction", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "transaction_id": txid, "customer_id": customer_id,
            "amount": amount, "hour": hour, "day_of_week": dow,
            "is_weekend": int(dow >= 5), "distance_km": distance,
            "device_age_days": device_age, "account_age_days": account_age,
            "transactions_last_24h": tx24, "avg_amount_30d": avg30,
            "failed_attempts_24h": failed, "is_new_device": int(new_device),
            "is_new_location": int(new_location),
            "merchant_risk": merchant, "channel": channel
        }])
        X = prepare_features(row)
        fraud_prob = float(model.predict_proba(X)[:, 1][0])
        anomaly_prob = float(anomaly_probability(iso, X)[0])
        risk_score, level = calculate_risk(fraud_prob, anomaly_prob, row.iloc[0])
        reason = explain(row.iloc[0], fraud_prob)

        a, b, c = st.columns(3)
        a.metric("Fraud probability", f"{fraud_prob:.1%}")
        b.metric("Anomaly probability", f"{anomaly_prob:.1%}")
        c.metric("Risk score", f"{risk_score}/100")

        if level in ["HIGH", "CRITICAL"]:
            st.error(f"🚨 {level} RISK — Review this transaction")
            save_alert({
                "transaction_id": txid, "customer_id": customer_id,
                "fraud_probability": fraud_prob, "anomaly_score": anomaly_prob,
                "risk_score": risk_score, "risk_level": level, "reason": reason
            })
        else:
            st.success(f"✅ {level} RISK")

        st.info(f"Primary risk signals: {reason}")

elif page == "Fraud Analytics":
    st.title("📊 Fraud Analytics")
    work = df.copy()
    work["predicted_probability"] = model.predict_proba(prepare_features(work))[:, 1]

    fig = px.scatter(
        work.sample(min(7000, len(work)), random_state=42),
        x="amount", y="predicted_probability", color="fraud",
        hover_data=["transaction_id", "customer_id", "channel"],
        title="Transaction Amount vs Fraud Probability"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top model features")
    importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)
    st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h"),
                    use_container_width=True)

elif page == "Customer Risk":
    st.title("👤 Customer Risk Profiling")
    X = prepare_features(df)
    df2 = df.copy()
    df2["fraud_probability"] = model.predict_proba(X)[:, 1]
    customer = df2.groupby("customer_id").agg(
        transactions=("transaction_id", "count"),
        avg_amount=("amount", "mean"),
        max_fraud_probability=("fraud_probability", "max"),
        avg_fraud_probability=("fraud_probability", "mean"),
        new_device_rate=("is_new_device", "mean"),
        new_location_rate=("is_new_location", "mean"),
        failed_attempts=("failed_attempts_24h", "sum")
    ).reset_index()
    customer["customer_risk"] = (
        .55 * customer["max_fraud_probability"] +
        .20 * customer["avg_fraud_probability"] +
        .10 * customer["new_device_rate"] +
        .10 * customer["new_location_rate"] +
        .05 * np.clip(customer["failed_attempts"] / 20, 0, 1)
    ) * 100
    customer = customer.sort_values("customer_risk", ascending=False)

    st.dataframe(customer.head(30), use_container_width=True)
    st.plotly_chart(
        px.bar(customer.head(15), x="customer_id", y="customer_risk",
               title="Highest-Risk Customers"),
        use_container_width=True
    )

elif page == "Alerts":
    st.title("🚨 Alert & Case Management")
    alerts = get_alerts()
    if alerts.empty:
        st.info("No alerts yet. Score a high-risk transaction from Live Transaction.")
    else:
        st.dataframe(alerts, use_container_width=True)
        open_alerts = alerts[alerts.status == "OPEN"]
        if not open_alerts.empty:
            selected = st.selectbox("Select alert to mark reviewed", open_alerts["id"].tolist())
            if st.button("Mark as Reviewed"):
                close_alert(int(selected))
                st.rerun()

elif page == "Drift Monitoring":
    st.title("📈 Model/Data Drift Monitoring")
    ref_path = MODEL_DIR / "reference_features.csv"
    reference = pd.read_csv(ref_path)
    current = prepare_features(df.sample(min(10000, len(df)), random_state=7))
    table = drift_table(reference, current)

    st.write("PSI interpretation: < 0.10 = OK, 0.10–0.25 = WATCH, > 0.25 = DRIFT.")
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(
        px.bar(table.head(15), x="PSI", y="feature", color="status",
               orientation="h", title="Highest Feature Drift"),
        use_container_width=True
    )
