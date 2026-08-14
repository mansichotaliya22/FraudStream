"""
FraudStream - Exploratory Data Analysis
Step 15: Visualize the dataset to understand patterns.

Run: python src/eda.py
Output: data/processed/eda_plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Config ──────────────────────────────────────────────
DATA_PATH = os.path.join("data", "raw", "creditcard.csv")
OUTPUT_DIR = os.path.join("data", "processed", "eda_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAUD_COLOR  = "#e74c3c"
NORMAL_COLOR = "#2ecc71"
plt.style.use("seaborn-v0_8-whitegrid")


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    target = next(c for c in df.columns if c.lower() in ("class", "isfraud"))
    fraud  = df[df[target] == 1]
    normal = df[df[target] == 0]
    print(f"  Normal : {len(normal):,}")
    print(f"  Fraud  : {len(fraud):,}\n")
    return df, target, fraud, normal


# ── Plot 1: Class Distribution ───────────────────────────
def plot_class_distribution(df, target):
    print("[1/5] Class distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution — Fraud vs Normal", fontsize=14, fontweight="bold")

    counts = df[target].value_counts().sort_index()
    labels = ["Normal", "Fraud"]
    colors = [NORMAL_COLOR, FRAUD_COLOR]

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Transaction Counts")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 500,
                     f"{val:,}", ha="center", fontweight="bold", fontsize=11)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=[f"{l}\n({v:,})" for l, v in zip(labels, counts.values)],
        colors=colors,
        autopct="%1.3f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[1].set_title("Class Proportions")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Plot 2: Amount Distribution ──────────────────────────
def plot_amount_distribution(fraud, normal):
    print("[2/5] Amount distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount: Fraud vs Normal", fontsize=14, fontweight="bold")

    # Linear scale
    axes[0].hist(normal["Amount"], bins=80, alpha=0.5, color=NORMAL_COLOR,
                 label=f"Normal (n={len(normal):,})", density=True)
    axes[0].hist(fraud["Amount"],  bins=80, alpha=0.8, color=FRAUD_COLOR,
                 label=f"Fraud  (n={len(fraud):,})",  density=True)
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Linear Scale")
    axes[0].set_xlim(0, normal["Amount"].quantile(0.99))
    axes[0].legend()

    # Log scale
    axes[1].hist(normal["Amount"].clip(lower=0.01), bins=80, alpha=0.5,
                 color=NORMAL_COLOR, label="Normal", density=True)
    axes[1].hist(fraud["Amount"].clip(lower=0.01),  bins=80, alpha=0.8,
                 color=FRAUD_COLOR,  label="Fraud",  density=True)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Amount ($) — Log Scale")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log Scale (reveals shape)")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_amount_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Plot 3: Time Analysis ────────────────────────────────
def plot_time_analysis(df, target, fraud, normal):
    print("[3/5] Time analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Time Analysis", fontsize=14, fontweight="bold")

    # Convert seconds to hours
    normal_h = (normal["Time"] / 3600) % 48
    fraud_h  = (fraud["Time"]  / 3600) % 48

    axes[0].hist(normal_h, bins=48, alpha=0.5, color=NORMAL_COLOR,
                 label="Normal", density=True)
    axes[0].hist(fraud_h,  bins=48, alpha=0.8, color=FRAUD_COLOR,
                 label="Fraud",  density=True)
    axes[0].set_xlabel("Hours since start")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Transaction Frequency Over Time")
    axes[0].legend()

    # Amount vs time scatter
    sample_n = normal.sample(min(5000, len(normal)), random_state=42)
    axes[1].scatter((sample_n["Time"] / 3600) % 48, sample_n["Amount"],
                    alpha=0.1, color=NORMAL_COLOR, s=5, label="Normal")
    axes[1].scatter(fraud_h, fraud["Amount"],
                    alpha=0.7, color=FRAUD_COLOR, s=20, label="Fraud")
    axes[1].set_xlabel("Hours since start")
    axes[1].set_ylabel("Amount ($)")
    axes[1].set_title("Amount vs Time")
    axes[1].set_ylim(0, df["Amount"].quantile(0.99))
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_time_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Plot 4: Feature Correlations with Fraud ──────────────
def plot_correlations(df, target):
    print("[4/5] Feature correlations...")
    v_cols = [c for c in df.columns if c.startswith("V")]
    corr   = df[v_cols + [target]].corr()[target].drop(target).sort_values()

    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = [FRAUD_COLOR if v > 0 else NORMAL_COLOR for v in corr.values]
    corr.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_title("Correlation of V Features with Fraud Label", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Pearson Correlation")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticklabels(corr.index, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_feature_correlations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")
    return corr


# ── Plot 5: Top Features Box Plots ───────────────────────
def plot_top_features(df, target, corr, fraud, normal):
    print("[5/5] Top discriminating features...")
    top = corr.abs().sort_values(ascending=False).head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Top 6 Features Most Correlated with Fraud", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    sample_n  = normal.sample(min(3000, len(normal)), random_state=42)
    combined  = pd.concat([
        sample_n.assign(Label="Normal"),
        fraud.assign(Label="Fraud")
    ])

    for i, feat in enumerate(top):
        sns.boxplot(
            data=combined, x="Label", y=feat,
            palette={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
            ax=axes[i], showfliers=False
        )
        axes[i].set_title(f"{feat}  (|corr|={corr.abs()[feat]:.3f})")
        axes[i].set_xlabel("")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Summary ──────────────────────────────────────────────
def print_summary(corr):
    top3_pos = corr.sort_values(ascending=False).head(3).index.tolist()
    top3_neg = corr.sort_values().head(3).index.tolist()
    print(f"""
─────────────────────────────────────────────────────
EDA SUMMARY
─────────────────────────────────────────────────────
  Features positively correlated with fraud : {top3_pos}
  Features negatively correlated with fraud : {top3_neg}

  Key observations:
  • 578:1 class imbalance → must use class_weight or SMOTE
  • Amount is right-skewed → apply log scaling
  • V features already PCA-scaled → no extra scaling needed
  • Some V features strongly separate fraud from normal

  ✅ EDA complete. Ready for Milestone 2 — Preprocessing.
─────────────────────────────────────────────────────
    """)


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  FRAUDSTREAM — EDA")
    print("=" * 55 + "\n")

    df, target, fraud, normal = load_data()

    plot_class_distribution(df, target)
    plot_amount_distribution(fraud, normal)
    plot_time_analysis(df, target, fraud, normal)
    corr = plot_correlations(df, target)
    plot_top_features(df, target, corr, fraud, normal)

    print_summary(corr)