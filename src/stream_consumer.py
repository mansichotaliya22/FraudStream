"""
FraudStream - Python Stream Consumer
Step 32: Read from Kafka, predict fraud, store results.

This replaces Spark for lightweight real-time processing.
Does exactly the same job at our transaction rate (5-20/sec).

Run: python src/stream_consumer.py
"""

import json
import time
import pickle
import os
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer

from db import (
    insert_transaction,
    insert_prediction,
    update_customer_summary,
    test_connection,
)
from redis_state import (
    get_customer_state,
    update_customer_state,
    extract_behavioral_features,
)

# ── Config ───────────────────────────────────────────────
KAFKA_BROKER  = "localhost:9092"
KAFKA_TOPIC   = "transactions"
KAFKA_GROUP   = "fraudstream-consumer"
MODEL_PATH    = os.path.join("models", "best_model.pkl")
SCALER_PATH   = os.path.join("models", "preprocessor.pkl")
FRAUD_THRESHOLD = 0.5   # probability above this = FRAUD

# V features the model was trained on
V_FEATURES = [f"V{i}" for i in range(1, 29)]


# ── Load Model ────────────────────────────────────────────
def load_model():
    print("  Loading fraud model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"  ✅ Model loaded: {type(model).__name__}")
    return model


def load_scaler():
    print("  Loading scaler...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"  ✅ Scaler loaded")
    return scaler


# ── Feature Engineering ───────────────────────────────────
def build_feature_vector(tx: dict, scaler) -> np.ndarray:
    """
    Build a feature vector for the fraud model.

    The model was trained on:
    - V1-V28 (PCA features from creditcard.csv)
    - Amount (scaled)
    - Time (scaled)

    For real-time transactions we don't have V features.
    So we use behavioral features from Redis instead,
    mapped to the same shape the model expects.

    Strategy:
    - Use amount and behavioral signals
    - Fill V features with 0 (neutral/normal signal)
    - Amount gets scaled same way as training
    """
    # Get customer state from Redis
    state    = get_customer_state(tx["customer_id"])
    behavior = extract_behavioral_features(tx, state)

    # Build feature array matching training shape (30 features)
    # V1-V28 = 0 (no PCA data for synthetic transactions)
    v_features = [0.0] * 28

   # Inject behavioral signals more aggressively
    v_features[13]  = -behavior["amount_deviation"] * 3      # V14
    v_features[16]  = -behavior["high_amount_flag"] * 5      # V17
    v_features[11]  = -behavior["new_location"] * 4          # V12
    v_features[9]   = -behavior["new_device"] * 4            # V10
    v_features[3]   =  behavior["tx_frequency"] * 1.5        # V4
    v_features[6]   = -behavior["amount_deviation"] * 2      # V7
    v_features[1]   = -behavior["high_amount_flag"] * 3      # V2
    # Scale Amount and Time same as training
    amount_time = scaler.transform([[tx["amount"], 0]])[0]
    scaled_amount = amount_time[0]
    scaled_time   = amount_time[1]

    # Final feature vector: [Time, V1..V28, Amount] = 30 features
    feature_vector = [scaled_time] + v_features + [scaled_amount]

    return np.array(feature_vector).reshape(1, -1)


# ── Predict ───────────────────────────────────────────────
def predict(model, feature_vector: np.ndarray):
    """Run fraud prediction and return probability + label."""
    fraud_prob  = model.predict_proba(feature_vector)[0][1]
    prediction  = "FRAUD" if fraud_prob >= FRAUD_THRESHOLD else "NORMAL"
    return round(float(fraud_prob), 4), prediction


# ── Process One Transaction ───────────────────────────────
def process_transaction(tx: dict, model, scaler):
    """Full pipeline for one transaction."""
    start_time = time.time()

    # 1. Build features
    features = build_feature_vector(tx, scaler)

    # 2. Predict
    fraud_prob, prediction = predict(model, features)

    # 3. Processing time
    proc_time = time.time() - start_time

    # 4. Store in PostgreSQL
    insert_transaction(tx)
    insert_prediction(
        transaction_id    = tx["transaction_id"],
        fraud_probability = fraud_prob,
        prediction        = prediction,
        processing_time   = proc_time,
    )

    # 5. Update customer summary in PostgreSQL
    update_customer_summary(
        customer_id = tx["customer_id"],
        amount      = tx["amount"],
        location    = tx["location"],
        device_id   = tx["device_id"],
        is_fraud    = 1 if prediction == "FRAUD" else 0,
    )

    # 6. Update customer state in Redis
    update_customer_state(tx["customer_id"], tx)

    return fraud_prob, prediction, proc_time


# ── Print Result ──────────────────────────────────────────
def print_result(tx: dict, fraud_prob: float,
                 prediction: str, proc_time: float):
    prob_pct = fraud_prob * 100
    amount   = tx["amount"]
    cid      = tx["customer_id"]
    loc      = tx["location"]
    txid     = tx["transaction_id"]

    if prediction == "FRAUD":
        print(f"  🚨 FRAUD  | {txid} | {cid} | "
              f"₹{amount:>10,.2f} | {loc:<12} | "
              f"prob={prob_pct:5.1f}% | {proc_time*1000:.1f}ms")
    else:
        print(f"  ✅ Normal | {txid} | {cid} | "
              f"₹{amount:>10,.2f} | {loc:<12} | "
              f"prob={prob_pct:5.1f}% | {proc_time*1000:.1f}ms")


# ── Main Consumer Loop ────────────────────────────────────
def run_consumer():
    print("=" * 60)
    print("  FRAUDSTREAM — STREAM CONSUMER")
    print("=" * 60)
    print(f"\n  Kafka  : {KAFKA_BROKER}")
    print(f"  Topic  : {KAFKA_TOPIC}")
    print(f"  Group  : {KAFKA_GROUP}")
    print(f"  Threshold : {FRAUD_THRESHOLD}\n")

    # Check DB connection
    print("  Checking database...")
    if not test_connection():
        print("  ❌ Cannot connect to PostgreSQL. Is Docker running?")
        return

    # Load model and scaler
    model  = load_model()
    scaler = load_scaler()

    # Connect to Kafka
    print(f"\n  Connecting to Kafka...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers    = KAFKA_BROKER,
        group_id             = KAFKA_GROUP,
        auto_offset_reset    = "latest",
        value_deserializer   = lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms  = -1,   # run forever
    )
    print(f"  ✅ Connected. Waiting for transactions...\n")
    print(f"  {'─'*58}")

    # Stats
    total   = 0
    frauds  = 0

    try:
        for message in consumer:
            tx = message.value

            fraud_prob, prediction, proc_time = process_transaction(
                tx, model, scaler
            )
            print_result(tx, fraud_prob, prediction, proc_time)

            total  += 1
            frauds += 1 if prediction == "FRAUD" else 0

            # Print summary every 20 transactions
            if total % 20 == 0:
                rate = frauds / total * 100
                print(f"\n  {'─'*58}")
                print(f"  📊 Stats: {total} processed | "
                      f"{frauds} fraud ({rate:.1f}%) | "
                      f"running avg proc time ≈ {proc_time*1000:.1f}ms")
                print(f"  {'─'*58}\n")

    except KeyboardInterrupt:
        print(f"\n  {'─'*58}")
        print(f"  Stopped by user.")
        print(f"  Total processed : {total}")
        print(f"  Fraud detected  : {frauds}")
        print(f"  Normal          : {total - frauds}")
        consumer.close()


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    run_consumer()