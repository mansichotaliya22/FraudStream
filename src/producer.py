"""
FraudStream - Transaction Generator + Kafka Producer
Step 26: Generate synthetic transactions and publish to Kafka.

Run:
  python src/producer.py --rate 5 --mode mixed
  python src/producer.py --rate 2 --mode fraud
  python src/producer.py --rate 10 --mode normal
"""

import json
import time
import random
import argparse
import uuid
from datetime import datetime
from kafka import KafkaProducer

# ── Config ───────────────────────────────────────────────
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC  = "transactions"

# ── Realistic Data Pools ─────────────────────────────────
CUSTOMERS = [f"C{str(i).zfill(3)}" for i in range(1, 51)]   # C001–C050
DEVICES   = [f"D{str(i).zfill(3)}" for i in range(1, 31)]   # D001–D030

LOCATIONS = [
    "Mumbai", "Delhi", "Pune", "Bangalore", "Chennai",
    "Hyderabad", "Kolkata", "Ahmedabad", "Jaipur", "Surat"
]

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "fuel",
    "clothing", "pharmacy", "travel", "entertainment",
    "utilities", "online_shopping"
]

PAYMENT_METHODS = ["UPI", "credit_card", "debit_card", "netbanking", "wallet"]

# ── Customer Profiles (normal behavior) ──────────────────
# Each customer has a home location and usual device
CUSTOMER_PROFILES = {
    cid: {
        "home_location" : random.choice(LOCATIONS),
        "usual_device"  : random.choice(DEVICES),
        "avg_amount"    : random.uniform(200, 3000),
    }
    for cid in CUSTOMERS
}


# ── Transaction Generators ────────────────────────────────

def normal_transaction(customer_id):
    """Generate a realistic normal transaction."""
    profile = CUSTOMER_PROFILES[customer_id]
    amount  = abs(random.gauss(profile["avg_amount"], profile["avg_amount"] * 0.3))
    amount  = round(min(amount, profile["avg_amount"] * 2), 2)

    return {
        "transaction_id"    : f"TX{uuid.uuid4().hex[:8].upper()}",
        "customer_id"       : customer_id,
        "amount"            : amount,
        "timestamp"         : datetime.now().isoformat(),
        "merchant_category" : random.choice(MERCHANT_CATEGORIES),
        "location"          : profile["home_location"],
        "payment_method"    : random.choice(PAYMENT_METHODS),
        "device_id"         : profile["usual_device"],
        "is_fraud"          : 0,
        "fraud_scenario"    : "none",
    }


def fraud_transaction(customer_id):
    """Generate a suspicious/fraudulent transaction."""
    profile  = CUSTOMER_PROFILES[customer_id]
    scenario = random.choice([
        "high_amount",
        "new_device",
        "new_location",
        "unusual_time",
        "combined",
    ])

    # Start with normal base
    tx = normal_transaction(customer_id)
    tx["is_fraud"]       = 1
    tx["fraud_scenario"] = scenario

    if scenario == "high_amount":
        # 10x to 50x normal amount
        tx["amount"] = round(profile["avg_amount"] * random.uniform(10, 50), 2)

    elif scenario == "new_device":
        # Device never seen before
        tx["device_id"] = f"D{random.randint(900, 999)}"

    elif scenario == "new_location":
        # Foreign location
        foreign = ["London", "Dubai", "Singapore", "New York", "Tokyo"]
        tx["location"] = random.choice(foreign)

    elif scenario == "unusual_time":
        # Transaction at 2–4 AM
        hour   = random.randint(2, 4)
        minute = random.randint(0, 59)
        ts     = datetime.now().replace(hour=hour, minute=minute)
        tx["timestamp"] = ts.isoformat()
        tx["amount"]    = round(profile["avg_amount"] * random.uniform(3, 8), 2)

    elif scenario == "combined":
        # Multiple red flags at once
        tx["amount"]    = round(profile["avg_amount"] * random.uniform(15, 40), 2)
        tx["device_id"] = f"D{random.randint(900, 999)}"
        foreign         = ["London", "Dubai", "Singapore", "New York", "Tokyo"]
        tx["location"]  = random.choice(foreign)

    return tx


def generate_transaction(mode):
    """Pick a customer and generate transaction based on mode."""
    customer_id = random.choice(CUSTOMERS)

    if mode == "normal":
        return normal_transaction(customer_id)
    elif mode == "fraud":
        return fraud_transaction(customer_id)
    elif mode == "mixed":
        # ~5% fraud rate — realistic
        if random.random() < 0.05:
            return fraud_transaction(customer_id)
        else:
            return normal_transaction(customer_id)


# ── Kafka Producer ────────────────────────────────────────

def create_producer():
    print(f"  Connecting to Kafka at {KAFKA_BROKER}...")
    producer = KafkaProducer(
        bootstrap_servers = KAFKA_BROKER,
        value_serializer  = lambda v: json.dumps(v).encode("utf-8"),
        key_serializer    = lambda k: k.encode("utf-8"),
    )
    print(f"  ✅ Connected to Kafka")
    return producer


def run_producer(rate, mode):
    print("=" * 55)
    print("  FRAUDSTREAM — TRANSACTION GENERATOR")
    print("=" * 55)
    print(f"\n  Mode  : {mode}")
    print(f"  Rate  : {rate} transactions/second")
    print(f"  Topic : {KAFKA_TOPIC}")
    print(f"\n  Press Ctrl+C to stop.\n")

    producer  = create_producer()
    interval  = 1.0 / rate
    count     = 0
    fraud_count = 0

    try:
        while True:
            tx = generate_transaction(mode)

            producer.send(
                KAFKA_TOPIC,
                key   = tx["customer_id"],
                value = tx,
            )

            count += 1
            if tx["is_fraud"] == 1:
                fraud_count += 1
                print(f"  🚨 FRAUD  | {tx['transaction_id']} | "
                      f"{tx['customer_id']} | "
                      f"₹{tx['amount']:>10,.2f} | "
                      f"{tx['location']:<12} | "
                      f"{tx['fraud_scenario']}")
            else:
                print(f"  ✅ Normal | {tx['transaction_id']} | "
                      f"{tx['customer_id']} | "
                      f"₹{tx['amount']:>10,.2f} | "
                      f"{tx['location']:<12}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n{'─'*55}")
        print(f"  Stopped.")
        print(f"  Total sent  : {count}")
        print(f"  Fraud sent  : {fraud_count}")
        print(f"  Normal sent : {count - fraud_count}")
        producer.flush()
        producer.close()


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudStream Transaction Generator")
    parser.add_argument("--rate", type=int,   default=2,
                        help="Transactions per second (default: 2)")
    parser.add_argument("--mode", type=str,   default="mixed",
                        choices=["normal", "fraud", "mixed"],
                        help="Transaction mode (default: mixed)")
    args = parser.parse_args()

    run_producer(args.rate, args.mode)