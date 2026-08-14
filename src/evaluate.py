"""
FraudStream - Model Evaluation
Step 20: Generate ROC curves, PR curves and full report.

Run: python src/evaluate.py
Output: data/processed/eval_plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# ── Config ───────────────────────────────────────────────
TEST_PATH  = os.path.join("data", "processed", "test.csv")
MODELS_DIR = "models"
OUTPUT_DIR = os.path.join("data", "processed", "eval_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAUD_COLOR  = "#e74c3c"
NORMAL_COLOR = "#2ecc71"
plt.style.use("seaborn-v0_8-whitegrid")

MODELS = {
    "Logistic Regression" : "logistic_regression.pkl",
    "Random Forest"       : "random_forest.pkl",
    "XGBoost"             : "xgboost.pkl",
}


# ── Load ─────────────────────────────────────────────────
def load_test_data():
    print("=" * 55)
    print("  FRAUDSTREAM — EVALUATION")
    print("=" * 55)
    test   = pd.read_csv(TEST_PATH)
    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]
    print(f"\n  Test rows : {len(X_test):,}")
    print(f"  Fraud     : {y_test.sum():,}")
    return X_test, y_test


def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Plot 1: ROC Curves (all models on one chart) ─────────
def plot_roc_curves(X_test, y_test):
    print("\n[1/4] ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for (name, filename), color in zip(MODELS.items(), colors):
        model   = load_model(filename)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc     = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(OUTPUT_DIR, "01_roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Plot 2: PR Curves (all models on one chart) ──────────
def plot_pr_curves(X_test, y_test):
    print("[2/4] PR curves...")
    fig, ax = plt.subplots(figsize=(8, 6))

    baseline = y_test.mean()
    colors   = ["#3498db", "#2ecc71", "#e74c3c"]

    for (name, filename), color in zip(MODELS.items(), colors):
        model   = load_model(filename)
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc   = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name}  (PR-AUC = {pr_auc:.4f})")

    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Random classifier ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(OUTPUT_DIR, "02_pr_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Plot 3: Confusion Matrices ───────────────────────────
def plot_confusion_matrices(X_test, y_test):
    print("[3/4] Confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (name, filename) in zip(axes, MODELS.items()):
        model  = load_model(filename)
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"])
        ax.set_yticklabels(["Normal", "Fraud"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black",
                        fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Plot 4: Fraud Probability Distribution ───────────────
def plot_probability_distribution(X_test, y_test):
    print("[4/4] Fraud probability distribution...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Fraud Probability Distribution", fontsize=14, fontweight="bold")

    for ax, (name, filename) in zip(axes, MODELS.items()):
        model   = load_model(filename)
        y_proba = model.predict_proba(X_test)[:, 1]

        ax.hist(y_proba[y_test == 0], bins=50, alpha=0.6,
                color=NORMAL_COLOR, label="Normal", density=True)
        ax.hist(y_proba[y_test == 1], bins=50, alpha=0.8,
                color=FRAUD_COLOR,  label="Fraud",  density=True)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Fraud Probability")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_probability_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ── Full Text Report ─────────────────────────────────────
def print_full_report(X_test, y_test):
    print(f"\n{'='*55}")
    print("  FULL CLASSIFICATION REPORT")
    print(f"{'='*55}")

    for name, filename in MODELS.items():
        model   = load_model(filename)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n── {name} ──────────────────────────────")
        print(classification_report(y_test, y_pred,
              target_names=["Normal", "Fraud"],
              zero_division=0))

        pr_auc  = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"  PR-AUC  : {pr_auc:.4f}")
        print(f"  ROC-AUC : {roc_auc:.4f}")


# ── Best Model Summary ───────────────────────────────────
def best_model_summary(X_test, y_test):
    print(f"\n{'='*55}")
    print("  BEST MODEL — XGBOOST SUMMARY")
    print(f"{'='*55}")

    model   = load_model("xgboost.pkl")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"""
  True  Positives (fraud caught)       : {tp}
  False Positives (innocent blocked)   : {fp}
  False Negatives (fraud missed)       : {fn}
  True  Negatives (normal correct)     : {tn:,}

  Precision : {precision_score(y_test, y_pred):.4f}
  Recall    : {recall_score(y_test, y_pred):.4f}
  F1        : {f1_score(y_test, y_pred):.4f}
  PR-AUC    : {average_precision_score(y_test, y_proba):.4f}
  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}

  In plain English:
  → Caught {tp} out of {tp+fn} fraud cases ({tp/(tp+fn)*100:.1f}% recall)
  → Only blocked {fp} innocent customers
  → Missed {fn} fraudulent transactions

  ✅ Evaluation complete.
  ✅ Plots saved to: data/processed/eval_plots/
  Next → Step 21: Commit to GitHub then build transaction generator
    """)


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    X_test, y_test = load_test_data()

    plot_roc_curves(X_test, y_test)
    plot_pr_curves(X_test, y_test)
    plot_confusion_matrices(X_test, y_test)
    plot_probability_distribution(X_test, y_test)

    print_full_report(X_test, y_test)
    best_model_summary(X_test, y_test)