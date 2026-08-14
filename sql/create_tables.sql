-- FraudStream Database Schema

-- Table 1: Raw transactions
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    VARCHAR(20)     PRIMARY KEY,
    customer_id       VARCHAR(10)     NOT NULL,
    amount            NUMERIC(12, 2)  NOT NULL,
    timestamp         TIMESTAMP       NOT NULL,
    merchant_category VARCHAR(50),
    location          VARCHAR(100),
    payment_method    VARCHAR(30),
    device_id         VARCHAR(20),
    is_fraud          INTEGER         DEFAULT 0,
    fraud_scenario    VARCHAR(30)     DEFAULT 'none',
    created_at        TIMESTAMP       DEFAULT NOW()
);

-- Table 2: Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL          PRIMARY KEY,
    transaction_id    VARCHAR(20)     NOT NULL,
    fraud_probability NUMERIC(6, 4)   NOT NULL,
    prediction        VARCHAR(10)     NOT NULL,
    model_version     VARCHAR(50)     DEFAULT 'xgboost_v1',
    processing_time   NUMERIC(8, 4),
    created_at        TIMESTAMP       DEFAULT NOW()
);

-- Table 3: Customer summary
CREATE TABLE IF NOT EXISTS customer_summary (
    customer_id       VARCHAR(10)     PRIMARY KEY,
    avg_amount        NUMERIC(12, 2)  DEFAULT 0,
    last_location     VARCHAR(100),
    last_device       VARCHAR(20),
    transaction_count INTEGER         DEFAULT 0,
    fraud_count       INTEGER         DEFAULT 0,
    last_updated      TIMESTAMP       DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_transactions_customer
    ON transactions(customer_id);

CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
    ON transactions(timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction
    ON predictions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON predictions(created_at);