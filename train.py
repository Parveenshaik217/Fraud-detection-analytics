from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from features import prepare_features

BASE = Path(__file__).parent
DATA = BASE / "data" / "transactions.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    import subprocess, sys
    subprocess.check_call([sys.executable, str(BASE / "generate_data.py")])

df = pd.read_csv(DATA)
X = prepare_features(df)
y = df["fraud"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, stratify=y, random_state=42
)

scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

model = XGBClassifier(
    n_estimators=350,
    max_depth=5,
    learning_rate=.05,
    subsample=.85,
    colsample_bytree=.85,
    min_child_weight=3,
    reg_lambda=2,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=4
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
threshold = .50
pred = (proba >= threshold).astype(int)

metrics = {
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "pr_auc": float(average_precision_score(y_test, proba)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "fraud_rate": float(y.mean()),
    "threshold": threshold
}

iso = IsolationForest(
    n_estimators=250,
    contamination=min(.10, max(.01, y.mean())),
    random_state=42,
    n_jobs=4
)
iso.fit(X_train)

joblib.dump(model, MODEL_DIR / "fraud_model.joblib")
joblib.dump(iso, MODEL_DIR / "anomaly_model.joblib")
joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

# Training reference sample used by drift monitoring.
X_train.sample(min(10000, len(X_train)), random_state=42).to_csv(
    MODEL_DIR / "reference_features.csv", index=False
)

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nTraining complete.")
print(json.dumps(metrics, indent=2))
