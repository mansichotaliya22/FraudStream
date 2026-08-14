"""
FraudStream - Model Training
Milestone 3: Train and compare LR, Random Forest, XGBoost.

Run: python src/train_models.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (precision_score, recall_score,
                                     f1_score, roc_auc_score,
                                     average_precision_score,
                                     confusion_matrix,
                                     classification_report)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# ── Config ───────────────────────────────────────────────
TRAIN_PATH  = os.path.join("data", "processed", "train.csv")
TEST_PATH   = os.path.join("data", "processed", "test.csv")
MODELS_DIR  = "models"
MLFLOW_URI  = "mlruns"

os.makedirs(MODELS_DIR, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("FraudStream")

RANDOM_STATE = 42


# ── Load Data ────────────────────────────────────────────
def load_data():
    print("=" * 55)
    print("  FRAUDSTREAM — MODEL TRAINING")
    print("=" * 55)

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    target = "Class"
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test  = test.drop(columns=[target])
    y_test  = test[target]

    print(f"\n  Train : {X_train.shape[0]:,} rows")
    print(f"  Test  : {X_test.shape[0]:,} rows")
    print(f"  Features : {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


# ── Evaluate ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision" : precision_score(y_test, y_pred,  zero_division=0),
        "recall"    : recall_score(y_test, y_pred,     zero_division=0),
        "f1"        : f1_score(y_test, y_pred,         zero_division=0),
        "roc_auc"   : roc_auc_score(y_test, y_proba),
        "pr_auc"    : average_precision_score(y_test,  y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'─'*35}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:>10.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")

    return metrics, y_pred, y_proba


# ── Train + Log ──────────────────────────────────────────
def train_and_log(name, model, params, X_train, y_train, X_test, y_test):
    print(f"\n{'─'*55}")
    print(f"  Training: {name}")
    print(f"{'─'*55}")

    with mlflow.start_run(run_name=name):
        # Train
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start
        print(f"  ✅ Trained in {duration:.1f}s")

        # Evaluate
        metrics, y_pred, y_proba = evaluate(model, X_test, y_test, name)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_time_seconds", duration)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ','_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)
        print(f"  ✅ Model saved : {model_path}")

    return metrics


# ── Models ───────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, X_test, y_test):
    params = {
        "model"         : "LogisticRegression",
        "C"             : 1.0,
        "max_iter"      : 1000,
        "class_weight"  : "balanced",
        "solver"        : "lbfgs",
    }
    model = LogisticRegression(
        C            = params["C"],
        max_iter     = params["max_iter"],
        class_weight = params["class_weight"],
        solver       = params["solver"],
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    return train_and_log("Logistic Regression", model, params, X_train, y_train, X_test, y_test)


def train_random_forest(X_train, y_train, X_test, y_test):
    params = {
        "model"          : "RandomForest",
        "n_estimators"   : 100,
        "max_depth"      : 10,
        "class_weight"   : "balanced",
        "min_samples_leaf": 2,
    }
    model = RandomForestClassifier(
        n_estimators    = params["n_estimators"],
        max_depth       = params["max_depth"],
        class_weight    = params["class_weight"],
        min_samples_leaf= params["min_samples_leaf"],
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
    )
    return train_and_log("Random Forest", model, params, X_train, y_train, X_test, y_test)


def train_xgboost(X_train, y_train, X_test, y_test):
    # scale_pos_weight handles imbalance in XGBoost
    # it is set to ratio of normal to fraud
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\n  XGBoost scale_pos_weight = {scale_pos_weight:.1f}")

    params = {
        "model"            : "XGBoost",
        "n_estimators"     : 200,
        "max_depth"        : 6,
        "learning_rate"    : 0.1,
        "scale_pos_weight" : round(scale_pos_weight, 2),
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
    }
    model = xgb.XGBClassifier(
        n_estimators     = params["n_estimators"],
        max_depth        = params["max_depth"],
        learning_rate    = params["learning_rate"],
        scale_pos_weight = params["scale_pos_weight"],
        subsample        = params["subsample"],
        colsample_bytree = params["colsample_bytree"],
        random_state     = RANDOM_STATE,
        eval_metric      = "logloss",
        verbosity        = 0,
        n_jobs           = -1,
    )
    return train_and_log("XGBoost", model, params, X_train, y_train, X_test, y_test)


# ── Compare and pick best ────────────────────────────────
def compare_models(results):
    print(f"\n{'='*55}")
    print("  MODEL COMPARISON")
    print(f"{'='*55}")
    print(f"\n  {'Model':<25} {'F1':>8} {'PR-AUC':>8} {'Recall':>8} {'Precision':>10}")
    print(f"  {'─'*60}")

    best_model = None
    best_score = 0

    for name, metrics in results.items():
        f1        = metrics["f1"]
        pr_auc    = metrics["pr_auc"]
        recall    = metrics["recall"]
        precision = metrics["precision"]
        print(f"  {name:<25} {f1:>8.4f} {pr_auc:>8.4f} {recall:>8.4f} {precision:>10.4f}")

        # Pick best by PR-AUC (best metric for imbalanced data)
        if pr_auc > best_score:
            best_score = pr_auc
            best_model = name

    print(f"\n  ✅ Best model : {best_model}  (PR-AUC = {best_score:.4f})")
    return best_model


def save_best_model(best_model_name):
    import shutil
    src  = os.path.join(MODELS_DIR, f"{best_model_name.lower().replace(' ','_')}.pkl")
    dest = os.path.join(MODELS_DIR, "best_model.pkl")
    shutil.copy(src, dest)
    print(f"  ✅ Best model copied to: {dest}")


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    results = {}
    results["Logistic Regression"] = train_logistic_regression(X_train, y_train, X_test, y_test)
    results["Random Forest"]       = train_random_forest(X_train, y_train, X_test, y_test)
    results["XGBoost"]             = train_xgboost(X_train, y_train, X_test, y_test)

    best = compare_models(results)
    save_best_model(best)

    print(f"""
  Next steps:
  → View experiments : mlflow ui
  → Then open        : http://localhost:5000
  → Step 19          : python src/evaluate.py
    """)