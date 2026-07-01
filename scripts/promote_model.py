"""
scripts/promote_model.py
Sets the 'Production' alias on the latest model version in MLflow,
and copies the artifact to mlartifacts/production/ for the Docker API.

Run this AFTER training if you want to update the production alias.
The Docker API does not need this — it loads from mlartifacts/production/
directly. This script is optional and only updates the MLflow UI label.

Prerequisites:
    docker compose up mlflow -d    (Docker MLflow must be running)

Usage:
    python scripts/promote_model.py
"""

import os, sys, shutil
import mlflow
from mlflow import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "fraud-xgb-v1"
ALIAS      = "Production"
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROD_PATH  = os.path.join(ROOT, "mlartifacts", "production")


def promote() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"Cannot reach MLflow at {MLFLOW_URI}: {e}")
        print("Make sure Docker MLflow is running: docker compose up mlflow -d")
        sys.exit(1)

    if not versions:
        print(f"No versions found for '{MODEL_NAME}'. Run training first.")
        sys.exit(1)

    latest      = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    version_num = latest.version
    run_id      = latest.run_id

    print(f"\n  Model   : {MODEL_NAME}")
    print(f"  Version : {version_num}")
    print(f"  Run ID  : {run_id}")

    # Set alias in MLflow UI
    client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=version_num)
    print(f"\n  Alias '{ALIAS}' set on version {version_num}")
    print(f"  View at: http://localhost:5000/#/models/{MODEL_NAME}")

    # Copy artifact to production folder (API reads from here)
    run    = client.get_run(run_id)
    exp_id = run.info.experiment_id
    src    = os.path.join(ROOT, "mlartifacts", exp_id, run_id, "artifacts", "model")

    if os.path.exists(src):
        if os.path.exists(PROD_PATH):
            shutil.rmtree(PROD_PATH)
        shutil.copytree(src, PROD_PATH)
        print(f"  Model copied to: {PROD_PATH}")
    else:
        print(f"  Artifact path not found: {src}")
        print(f"  Using existing production folder (set by train.py).")

    print(f"\n  Run: docker compose up --build")


if __name__ == "__main__":
    promote()
