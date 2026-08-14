"""
FraudStream - Database Writer
Step 29: Insert transactions and predictions into PostgreSQL.

Imported by spark_consumer.py
"""

import psycopg2
import psycopg2.extras
import os
from datetime import datetime

# ── Config ───────────────────────────────────────────────
DB_CONFIG = {
    "host"     : "localhost",
    "port"     : 5432,
    "database" : "fraudstream",
    "user"     : "fraud_user",
    "password" : "fraud_pass",
}


def get_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)


# ── Insert Transaction ────────────────────────────────────
def insert_transaction(tx: dict):
    """Insert a raw transaction into the transactions table."""
    sql = """
        INSERT INTO transactions (
            transaction_id, customer_id, amount, timestamp,
            merchant_category, location, payment_method,
            device_id, is_fraud, fraud_scenario
        ) VALUES (
            %(transaction_id)s, %(customer_id)s, %(amount)s, %(timestamp)s,
            %(merchant_category)s, %(location)s, %(payment_method)s,
            %(device_id)s, %(is_fraud)s, %(fraud_scenario)s
        )
        ON CONFLICT (transaction_id) DO NOTHING;
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, tx)
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ insert_transaction error: {e}")
        return False


# ── Insert Prediction ─────────────────────────────────────
def insert_prediction(transaction_id: str,
                      fraud_probability: float,
                      prediction: str,
                      processing_time: float,
                      model_version: str = "xgboost_v1"):
    """Insert a model prediction into the predictions table."""
    sql = """
        INSERT INTO predictions (
            transaction_id, fraud_probability,
            prediction, processing_time, model_version
        ) VALUES (%s, %s, %s, %s, %s);
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (
            transaction_id,
            round(fraud_probability, 4),
            prediction,
            round(processing_time, 4),
            model_version,
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ insert_prediction error: {e}")
        return False


# ── Update Customer Summary ───────────────────────────────
def update_customer_summary(customer_id: str,
                             amount: float,
                             location: str,
                             device_id: str,
                             is_fraud: int):
    """Upsert customer summary after each transaction."""
    sql = """
        INSERT INTO customer_summary (
            customer_id, avg_amount, last_location,
            last_device, transaction_count, fraud_count, last_updated
        ) VALUES (%s, %s, %s, %s, 1, %s, NOW())
        ON CONFLICT (customer_id) DO UPDATE SET
            avg_amount        = (customer_summary.avg_amount *
                                  customer_summary.transaction_count + %s) /
                                  (customer_summary.transaction_count + 1),
            last_location     = EXCLUDED.last_location,
            last_device       = EXCLUDED.last_device,
            transaction_count = customer_summary.transaction_count + 1,
            fraud_count       = customer_summary.fraud_count + %s,
            last_updated      = NOW();
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (
            customer_id, amount, location,
            device_id, is_fraud,
            amount, is_fraud,
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ update_customer_summary error: {e}")
        return False


# ── Test Connection ───────────────────────────────────────
def test_connection():
    """Verify database connection is working."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"  ✅ PostgreSQL connected")
        print(f"     {version[:50]}")
        return True
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return False


# ── Quick Stats ───────────────────────────────────────────
def get_stats():
    """Get current counts from database."""
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM transactions;")
        tx_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM predictions;")
        pred_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'FRAUD';")
        fraud_count = cur.fetchone()[0]

        cur.close()
        conn.close()

        print(f"\n  Database Stats:")
        print(f"  Transactions : {tx_count:,}")
        print(f"  Predictions  : {pred_count:,}")
        print(f"  Fraud alerts : {fraud_count:,}")
        return tx_count, pred_count, fraud_count

    except Exception as e:
        print(f"  ❌ get_stats error: {e}")
        return 0, 0, 0


# ── Main (test when run directly) ────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  FRAUDSTREAM — DATABASE CONNECTION TEST")
    print("=" * 55 + "\n")

    if test_connection():
        # Insert a test transaction
        test_tx = {
            "transaction_id"    : "TX_TEST_001",
            "customer_id"       : "C001",
            "amount"            : 999.99,
            "timestamp"         : datetime.now().isoformat(),
            "merchant_category" : "grocery",
            "location"          : "Mumbai",
            "payment_method"    : "UPI",
            "device_id"         : "D001",
            "is_fraud"          : 0,
            "fraud_scenario"    : "none",
        }

        print("\n  Inserting test transaction...")
        insert_transaction(test_tx)

        print("  Inserting test prediction...")
        insert_prediction("TX_TEST_001", 0.0234, "NORMAL", 0.0123)

        print("  Updating customer summary...")
        update_customer_summary("C001", 999.99, "Mumbai", "D001", 0)

        get_stats()

        print("\n  ✅ Database test complete.")
        print("  ✅ Ready for Spark consumer.")