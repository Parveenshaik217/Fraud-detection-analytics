-- Optional SQL schema for a production database.
CREATE TABLE transactions (
    transaction_id VARCHAR(30) PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    hour INT,
    day_of_week INT,
    is_weekend BOOLEAN,
    distance_km DECIMAL(12,2),
    device_age_days INT,
    account_age_days INT,
    transactions_last_24h INT,
    avg_amount_30d DECIMAL(15,2),
    failed_attempts_24h INT,
    is_new_device BOOLEAN,
    is_new_location BOOLEAN,
    merchant_risk DECIMAL(5,4),
    channel VARCHAR(20),
    fraud BOOLEAN
);

CREATE TABLE fraud_alerts (
    id BIGINT PRIMARY KEY,
    transaction_id VARCHAR(30),
    customer_id BIGINT,
    fraud_probability DECIMAL(8,6),
    anomaly_score DECIMAL(8,6),
    risk_score DECIMAL(8,4),
    risk_level VARCHAR(20),
    reason TEXT,
    status VARCHAR(20),
    created_at TIMESTAMP
);
