"""
training/train.py
Trains a fraud detection model on the real Credit Card Fraud dataset.

Features : V1-V28 (PCA) + Amount = 29 features total
Pipeline  : RobustScaler -> XGBoost
Imbalance : scale_pos_weight ~578  (0.172% fraud rate)

Prerequisites:
    1. python data/download_dataset.py   (download dataset first)
    2. docker compose up mlflow -d       (start MLflow in Docker)

Usage:
    python training/train.py
"""

from __future__ import annotations
import os, sys, shutil, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "creditcard.csv")
PROD_PATH = os.path.join(ROOT, "mlartifacts", "production")

# ── Config ────────────────────────────────────────────────────────────────────
# MLflow runs inside Docker — training talks to it at localhost:5000
# Start it first with: docker compose up mlflow -d
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT   = "creditcard-fraud-v1"
MODEL_NAME   = "fraud-xgb-v1"
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
# ─────────────────────────────────────────────────────────────────────────────


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds 0.10-0.90, return the one maximising F1."""
    best_f1, best_t = 0.0, 0.5
    for t in [i / 100 for i in range(10, 91)]:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def save_to_production(pipeline: Pipeline, run_id: str, exp_id: str) -> None:
    """
    Copy the MLflow artifact folder to mlartifacts/production/.
    The Docker API container mounts mlartifacts/ as a volume and loads
    the model from /mlartifacts/production/ at startup — no registry
    query needed, which avoids the DNS rebinding error inside Docker.
    """
    src = os.path.join(ROOT, "mlartifacts", exp_id, run_id, "artifacts", "model")
    dst = PROD_PATH

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        log.info(f"Model copied to production: {dst}")
    else:
        # Fallback: save with joblib directly
        import joblib
        joblib.dump(pipeline, os.path.join(dst, "pipeline.pkl"))
        with open(os.path.join(dst, "MLmodel"), "w") as f:
            f.write("flavors:\n  sklearn:\n    pickled_model: pipeline.pkl\n")
        log.info(f"Model saved via joblib: {dst}")


def save_direct(pipeline: Pipeline) -> None:
    """Save pipeline without MLflow (fallback if MLflow is unreachable)."""
    import joblib
    os.makedirs(PROD_PATH, exist_ok=True)
    joblib.dump(pipeline, os.path.join(PROD_PATH, "pipeline.pkl"))
    with open(os.path.join(PROD_PATH, "MLmodel"), "w") as f:
        f.write("flavors:\n  sklearn:\n    pickled_model: pipeline.pkl\n")
    log.info(f"Model saved directly: {PROD_PATH}")


def train() -> None:
    print("\n" + "=" * 64)
    print("  FraudGuard — Training on Real Credit Card Fraud Dataset")
    print("=" * 64 + "\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found: {DATA_PATH}")
        print("Run first:  python data/download_dataset.py")
        sys.exit(1)

    log.info(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURE_COLS + ["Class"] if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print("Re-download: python data/download_dataset.py")
        sys.exit(1)

    X = df[FEATURE_COLS]
    y = df["Class"].astype(int)

    fraud_rate = y.mean()
    log.info(f"  {len(df):,} rows | fraud: {y.sum()} ({fraud_rate:.3%}) | legit: {(y==0).sum():,}")

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── scale_pos_weight for imbalanced classes ────────────────────────────────
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    log.info(f"  scale_pos_weight = {spw:.1f}")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    xgb_params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "scale_pos_weight": spw,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "eval_metric":      "aucpr",
        "random_state":     42,
        "n_jobs":           -1,
        "device":           "cpu",
    }

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  xgb.XGBClassifier(**xgb_params)),
    ])

    # ── Connect to MLflow (Docker) ─────────────────────────────────────────────
    mlflow_ok = True
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)
        log.info(f"MLflow connected: {MLFLOW_URI}")
    except Exception as e:
        log.warning(f"MLflow not reachable ({e}). Will save model directly.")
        mlflow_ok = False

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Training RobustScaler + XGBoost (300 trees) ...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_t, _ = find_best_threshold(np.array(y_test), y_proba)
    y_pred    = (y_proba >= best_t).astype(int)

    metrics = {
        "roc_auc":        round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc":         round(average_precision_score(y_test, y_proba), 4),
        "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":             round(f1_score(y_test, y_pred, zero_division=0), 4),
        "best_threshold": best_t,
        "fraud_rate":     round(float(fraud_rate), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Results at threshold {best_t:.2f}:")
    print(f"    ROC-AUC  : {metrics['roc_auc']}")
    print(f"    PR-AUC   : {metrics['pr_auc']}  <- key metric for imbalanced data")
    print(f"    Precision: {metrics['precision']}")
    print(f"    Recall   : {metrics['recall']}")
    print(f"    F1       : {metrics['f1']}")
    print(f"\n    Confusion Matrix:")
    print(f"                  Pred Legit  Pred Fraud")
    print(f"    True Legit  {cm[0][0]:>10,}  {cm[0][1]:>10,}")
    print(f"    True Fraud  {cm[1][0]:>10,}  {cm[1][1]:>10,}")

    # ── Log + save ────────────────────────────────────────────────────────────
    if mlflow_ok:
        try:
            with mlflow.start_run() as run:
                mlflow.log_params(xgb_params)
                mlflow.log_params({"feature_count": len(FEATURE_COLS), "scaler": "RobustScaler"})
                mlflow.log_metrics(metrics)

                from mlflow.models.signature import infer_signature
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                    signature=infer_signature(X_train, y_proba),
                    input_example=X_train.iloc[:3],
                )
                run_id = run.info.run_id
                exp_id = run.info.experiment_id

            print(f"\n  MLflow run logged: {run_id}")
            print(f"  MLflow UI: http://localhost:5000")
            save_to_production(pipeline, run_id, exp_id)

        except Exception as e:
            log.warning(f"MLflow log failed: {e}. Saving directly.")
            save_direct(pipeline)
    else:
        save_direct(pipeline)

    print(f"\n  Production model ready: {PROD_PATH}")
    print(f"\n{'='*64}")
    print(f"  Next step: docker compose up --build")
    print(f"  Dashboard: http://localhost:8000")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    train()
