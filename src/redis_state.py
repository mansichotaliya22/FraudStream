"""
FraudStream - Redis Customer State
Step 30: Store and retrieve customer behavioral state.

Why Redis and not PostgreSQL for this?
- Redis is in-memory → microsecond reads
- PostgreSQL is on disk → millisecond reads
- During streaming we need customer state on EVERY transaction
- At 10 transactions/second that is 10 DB reads/second minimum
- Redis handles this easily, PostgreSQL would become a bottleneck

Imported by spark_consumer.py
"""

import redis
import json
from datetime import datetime

# ── Config ───────────────────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0

# How long to keep customer state (7 days in seconds)
TTL_SECONDS = 7 * 24 * 60 * 60


def get_client():
    """Create and return a Redis client."""
    return redis.Redis(
        host     = REDIS_HOST,
        port     = REDIS_PORT,
        db       = REDIS_DB,
        decode_responses = True,
    )


# ── Get Customer State ────────────────────────────────────
def get_customer_state(customer_id: str) -> dict:
    """
    Retrieve customer behavioral state from Redis.
    Returns default state if customer is new.
    """
    client = get_client()
    key    = f"customer:{customer_id}"
    data   = client.get(key)

    if data is None:
        # New customer — return default state
        return {
            "customer_id"          : customer_id,
            "avg_amount"           : 0.0,
            "transaction_count"    : 0,
            "last_location"        : None,
            "last_device"          : None,
            "last_timestamp"       : None,
            "transactions_last_hour" : 0,
            "is_new_customer"      : True,
        }

    state = json.loads(data)
    state["is_new_customer"] = False
    return state


# ── Update Customer State ─────────────────────────────────
def update_customer_state(customer_id: str, tx: dict):
    """
    Update customer state after processing a transaction.
    Uses running average for amount.
    """
    client = get_client()
    key    = f"customer:{customer_id}"

    # Get existing state
    state  = get_customer_state(customer_id)

    # Update running average amount
    count      = state["transaction_count"]
    old_avg    = state["avg_amount"]
    new_amount = tx["amount"]
    new_avg    = ((old_avg * count) + new_amount) / (count + 1)

    # Update state
    new_state = {
        "customer_id"            : customer_id,
        "avg_amount"             : round(new_avg, 2),
        "transaction_count"      : count + 1,
        "last_location"          : tx["location"],
        "last_device"            : tx["device_id"],
        "last_timestamp"         : tx["timestamp"],
        "transactions_last_hour" : state.get("transactions_last_hour", 0) + 1,
    }

    client.setex(key, TTL_SECONDS, json.dumps(new_state))
    return new_state


# ── Feature Engineering from State ───────────────────────
def extract_behavioral_features(tx: dict, state: dict) -> dict:
    """
    Compare current transaction against customer history.
    These features are fed into the fraud model.
    """
    features = {}

    # Is this a new location for this customer?
    features["new_location"] = int(
        state["last_location"] is not None and
        tx["location"] != state["last_location"]
    )

    # Is this a new device?
    features["new_device"] = int(
        state["last_device"] is not None and
        tx["device_id"] != state["last_device"]
    )

    # How much does this deviate from their average?
    if state["avg_amount"] > 0:
        features["amount_deviation"] = round(
            tx["amount"] / state["avg_amount"], 4
        )
    else:
        features["amount_deviation"] = 1.0

    # Is this an unusually large transaction? (more than 5x average)
    features["high_amount_flag"] = int(
        state["avg_amount"] > 0 and
        tx["amount"] > state["avg_amount"] * 5
    )

    # Transaction frequency in last hour
    features["tx_frequency"] = state.get("transactions_last_hour", 0)

    # Is this a new customer (no history)?
    features["is_new_customer"] = int(state.get("is_new_customer", False))

    return features


# ── Test ─────────────────────────────────────────────────
def test_redis():
    print("=" * 55)
    print("  FRAUDSTREAM — REDIS TEST")
    print("=" * 55 + "\n")

    try:
        client = get_client()
        client.ping()
        print("  ✅ Redis connected\n")
    except Exception as e:
        print(f"  ❌ Redis connection failed: {e}")
        return

    # Simulate 3 transactions for customer C001
    test_transactions = [
        {
            "transaction_id"    : "TX_R001",
            "customer_id"       : "C001",
            "amount"            : 500.00,
            "timestamp"         : datetime.now().isoformat(),
            "location"          : "Mumbai",
            "device_id"         : "D001",
            "merchant_category" : "grocery",
            "payment_method"    : "UPI",
        },
        {
            "transaction_id"    : "TX_R002",
            "customer_id"       : "C001",
            "amount"            : 750.00,
            "timestamp"         : datetime.now().isoformat(),
            "location"          : "Mumbai",
            "device_id"         : "D001",
            "merchant_category" : "restaurant",
            "payment_method"    : "UPI",
        },
        {
            "transaction_id"    : "TX_R003",
            "customer_id"       : "C001",
            "amount"            : 25000.00,  # suspicious — high amount
            "timestamp"         : datetime.now().isoformat(),
            "location"          : "London",   # suspicious — new location
            "device_id"         : "D999",     # suspicious — new device
            "merchant_category" : "electronics",
            "payment_method"    : "credit_card",
        },
    ]

    for i, tx in enumerate(test_transactions):
        print(f"  Transaction {i+1}: ₹{tx['amount']:,.2f} | "
              f"{tx['location']} | {tx['device_id']}")

        # Get state BEFORE this transaction
        state    = get_customer_state(tx["customer_id"])
        features = extract_behavioral_features(tx, state)

        print(f"    State before  : avg=₹{state['avg_amount']:,.2f} | "
              f"count={state['transaction_count']}")
        print(f"    Features      : {features}")

        # Update state AFTER
        new_state = update_customer_state(tx["customer_id"], tx)
        print(f"    State after   : avg=₹{new_state['avg_amount']:,.2f} | "
              f"count={new_state['transaction_count']}\n")

    print("  ✅ Redis test complete.")
    print("  ✅ Ready for Spark consumer.")


if __name__ == "__main__":
    test_redis()