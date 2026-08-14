"""
FraudStream - Preprocessing Pipeline
Milestone 2: Clean, scale and split the dataset.

Run: python src/preprocessing.py
Output: data/processed/train.csv and test.csv
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ── Config ──────────────────────────────────────────────
DATA_PATH  = os.path.join("data", "raw", "creditcard.csv")
TRAIN_PATH = os.path.join("data", "processed", "train.csv")
TEST_PATH  = os.path.join("data", "processed", "test.csv")
SCALER_PATH = os.path.join("models", "preprocessor.pkl")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

TEST_SIZE    = 0.2
RANDOM_STATE = 42


def load_data():
    print("=" * 55)
    print("  FRAUDSTREAM — PREPROCESSING")
    print("=" * 55)
    print(f"\n📂 Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df):
    print("\n─── Step 1: Remove duplicates ───────────────────────")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"  Removed : {before - after:,} duplicate rows")
    print(f"  Remaining : {after:,} rows")
    return df


def separate_features_label(df):
    print("\n─── Step 2: Separate features and label ─────────────")

    # Auto detect target column
    target = next(c for c in df.columns if c.lower() in ("class", "isfraud"))
    print(f"  Target column : '{target}'")

    X = df.drop(columns=[target])
    y = df[target]

    print(f"  Features (X) shape : {X.shape}")
    print(f"  Label    (y) shape : {y.shape}")
    print(f"  Fraud cases        : {y.sum():,} ({y.mean()*100:.3f}%)")
    return X, y


def scale_features(X):
    print("\n─── Step 3: Scale Amount and Time ───────────────────")
    print("  V1–V28  : already PCA scaled — skipping")
    print("  Amount  : applying RobustScaler")
    print("  Time    : applying RobustScaler")

    scaler = RobustScaler()

    X = X.copy()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    print(f"  Amount after scaling — mean: {X['Amount'].mean():.4f}, std: {X['Amount'].std():.4f}")
    print(f"  Time   after scaling — mean: {X['Time'].mean():.4f}, std: {X['Time'].std():.4f}")

    # Save scaler so streaming pipeline can use same scaling
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  ✅ Scaler saved to: {SCALER_PATH}")

    return X, scaler


def split_data(X, y):
    print("\n─── Step 4: Train/Test Split ─────────────────────────")
    print(f"  Test size    : {TEST_SIZE*100:.0f}%")
    print(f"  Strategy     : Stratified (preserves fraud ratio)")
    print(f"  Random state : {RANDOM_STATE}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y        # ← keeps same fraud % in train and test
    )

    print(f"\n  Train set : {len(X_train):,} rows")
    print(f"    Normal  : {(y_train==0).sum():,}")
    print(f"    Fraud   : {(y_train==1).sum():,}")

    print(f"\n  Test set  : {len(X_test):,} rows")
    print(f"    Normal  : {(y_test==0).sum():,}")
    print(f"    Fraud   : {(y_test==1).sum():,}")

    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test):
    print("\n─── Step 5: Save train and test sets ────────────────")

    train_df = X_train.copy()
    train_df["Class"] = y_train.values

    test_df = X_test.copy()
    test_df["Class"] = y_test.values

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,  index=False)

    print(f"  ✅ Train saved : {TRAIN_PATH}  ({len(train_df):,} rows)")
    print(f"  ✅ Test  saved : {TEST_PATH}   ({len(test_df):,} rows)")


def summary(X_train, X_test, y_train, y_test):
    print(f"""
{"=" * 55}
  PREPROCESSING COMPLETE
{"=" * 55}

  Train rows : {len(X_train):,}
  Test rows  : {len(X_test):,}
  Features   : {X_train.shape[1]}

  Train fraud rate : {y_train.mean()*100:.3f}%
  Test  fraud rate : {y_test.mean()*100:.3f}%
  (These should be nearly identical — confirms stratification worked)

  Scaler saved : models/preprocessor.pkl
  Train saved  : data/processed/train.csv
  Test  saved  : data/processed/test.csv

  ✅ Ready for Milestone 3 — Model Training
{"=" * 55}
    """)


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    df                          = load_data()
    df                          = remove_duplicates(df)
    X, y                        = separate_features_label(df)
    X, scaler                   = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_splits(X_train, X_test, y_train, y_test)
    summary(X_train, X_test, y_train, y_test)