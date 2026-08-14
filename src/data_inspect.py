"""
FraudStream - Dataset Inspection
Step 14: Validate the raw dataset before touching it.

Run: python src/data_inspect.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Config ──────────────────────────────────────────────
DATA_PATH = os.path.join("data", "raw", "creditcard.csv")


def load_dataset(path):
    print("=" * 55)
    print("  FRAUDSTREAM — DATASET INSPECTION")
    print("=" * 55)

    if not os.path.exists(path):
        print(f"\n❌ File not found: {path}")
        print("   Place creditcard.csv inside data/raw/")
        sys.exit(1)

    print(f"\n📂 Loading: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded successfully.\n")
    return df


def check_shape(df):
    print("─" * 55)
    print("1. SHAPE")
    print("─" * 55)
    rows, cols = df.shape
    print(f"  Rows    : {rows:,}")
    print(f"  Columns : {cols}")
    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  Memory  : {size_mb:.1f} MB")


def check_columns(df):
    print("\n─" * 55)
    print("2. COLUMN NAMES AND TYPES")
    print("─" * 55)
    for col in df.columns:
        print(f"  {col:<15} {str(df[col].dtype):<10}")


def check_missing(df):
    print("\n─" * 55)
    print("3. MISSING VALUES")
    print("─" * 55)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  ✅ No missing values.")
    else:
        print(missing)


def check_duplicates(df):
    print("\n─" * 55)
    print("4. DUPLICATE ROWS")
    print("─" * 55)
    n = df.duplicated().sum()
    print(f"  Duplicate rows: {n:,}")
    if n > 0:
        print(f"  ⚠️  Will remove in preprocessing.")


def check_class_distribution(df):
    print("\n─" * 55)
    print("5. CLASS DISTRIBUTION")
    print("─" * 55)

    # Auto detect target column
    target = None
    for col in df.columns:
        if col.lower() in ("class", "isfraud", "fraud", "label"):
            target = col
            break

    if target is None:
        print("  ⚠️  Could not detect target column.")
        print(f"  Columns available: {list(df.columns)}")
        return

    print(f"  Target column : '{target}'")
    counts = df[target].value_counts().sort_index()
    total = len(df)

    for label, count in counts.items():
        pct = count / total * 100
        name = "Normal" if label == 0 else "Fraud "
        bar = "█" * int(pct / 2)
        print(f"  {name} ({label}): {count:>7,}  ({pct:6.3f}%)  {bar}")

    ratio = counts[0] / counts[1]
    print(f"\n  Imbalance ratio : {ratio:.0f}:1")
    print(f"  ⚠️  Accuracy is NOT a valid metric for this data.")
    print(f"  ✅ Use Precision, Recall, F1, PR-AUC instead.")


def check_amount(df):
    print("\n─" * 55)
    print("6. AMOUNT FEATURE")
    print("─" * 55)
    if "Amount" not in df.columns:
        print("  No Amount column found.")
        return
    print(f"  Min    : ${df['Amount'].min():.2f}")
    print(f"  Max    : ${df['Amount'].max():.2f}")
    print(f"  Mean   : ${df['Amount'].mean():.2f}")
    print(f"  Median : ${df['Amount'].median():.2f}")
    print(f"  Std    : ${df['Amount'].std():.2f}")


def check_sample(df):
    print("\n─" * 55)
    print("7. FIRST 3 ROWS (non-V columns only)")
    print("─" * 55)
    cols = [c for c in df.columns if not c.startswith("V")]
    print(df[cols].head(3).to_string(index=False))


def summary(df):
    print("\n" + "=" * 55)
    print("  INSPECTION COMPLETE")
    print("=" * 55)
    target = next((c for c in df.columns if c.lower() in ("class","isfraud")), None)
    fraud = df[df[target] == 1].shape[0] if target else "N/A"
    print(f"""
  Total rows     : {len(df):,}
  Total features : {df.shape[1] - 1}
  Fraud cases    : {fraud:,}
  Missing values : {df.isnull().sum().sum()}
  Duplicates     : {df.duplicated().sum():,}

  ✅ Dataset looks good. Ready for EDA.
  Next → python src/eda.py
    """)


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    check_shape(df)
    check_columns(df)
    check_missing(df)
    check_duplicates(df)
    check_class_distribution(df)
    check_amount(df)
    check_sample(df)
    summary(df)