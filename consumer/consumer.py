"""
consumer/consumer.py
Reads real transaction events from Kafka, calls the FastAPI scoring endpoint,
and passes the true_label from the real dataset so the dashboard can show
live accuracy metrics (TP, TN, FP, FN, Precision, Recall).
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
API_URL        = os.getenv("API_URL", "http://localhost:8000/predict")
INPUT_TOPIC    = "transactions.raw"
ALERT_TOPIC    = "fraud.alerts"
GROUP_ID       = "fraudguard-scorer-v1"
API_TIMEOUT_S  = 1.0
HEARTBEAT_N    = 500
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def wait_for_kafka(max_retries: int = 30, delay: int = 5) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            p = KafkaProducer(bootstrap_servers=KAFKA_SERVERS)
            p.close()
            log.info(f"✅  Kafka ready at {KAFKA_SERVERS}")
            return
        except NoBrokersAvailable:
            log.warning(f"Kafka not ready ({attempt}/{max_retries}) — retry in {delay}s ...")
            time.sleep(delay)
    raise RuntimeError("Kafka unavailable")


def wait_for_api(max_retries: int = 30, delay: int = 5) -> None:
    health_url = API_URL.rsplit("/predict", 1)[0] + "/health"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(health_url, timeout=3)
            if r.status_code == 200 and r.json().get("model_loaded"):
                log.info("✅  API ready — model loaded")
                return
            log.warning(f"API not ready ({attempt}/{max_retries}) — model_loaded=False")
        except Exception:
            log.warning(f"API not reachable ({attempt}/{max_retries}) — retry in {delay}s ...")
        time.sleep(delay)
    log.warning("API never became ready — starting anyway")


def build_features(txn: dict) -> dict:
    """Extract the 29 model features from the Kafka message."""
    return {col: float(txn.get(col, 0.0)) for col in FEATURE_COLS}


def run() -> None:
    wait_for_kafka()
    wait_for_api()

    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id=GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    alert_producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    counts = {"processed": 0, "fraud": 0, "errors": 0, "timeouts": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    log.info(f"🎧  Consuming '{INPUT_TOPIC}' → scoring via {API_URL}")

    for msg in consumer:
        txn    = msg.value
        txn_id = txn.get("transaction_id", "unknown")
        true_label = txn.get("true_label", None)

        try:
            features = build_features(txn)

            # Pass true_label as query param so the dashboard can track accuracy
            params = {"txn_id": txn_id}
            if true_label is not None:
                params["true_label"] = int(true_label)

            resp = requests.post(
                API_URL,
                params=params,
                json=features,
                timeout=API_TIMEOUT_S,
            )
            resp.raise_for_status()
            result = resp.json()

            counts["processed"] += 1
            is_fraud = result["is_fraud"]

            # Track confusion matrix
            if true_label is not None:
                tl = int(true_label)
                if is_fraud and tl == 1:   counts["tp"] += 1
                elif not is_fraud and tl == 0: counts["tn"] += 1
                elif is_fraud and tl == 0:  counts["fp"] += 1
                elif not is_fraud and tl == 1: counts["fn"] += 1

            if is_fraud:
                counts["fraud"] += 1
                alert = {
                    **txn,
                    "fraud_probability": result["fraud_probability"],
                    "latency_ms":        result["latency_ms"],
                    "alerted_at":        time.time(),
                }
                alert_producer.send(ALERT_TOPIC, value=alert)
                log.warning(
                    f"🚨 FRAUD  id={txn_id[:14]}  "
                    f"amount=${txn.get('Amount', 0):.2f}  "
                    f"true={true_label}  "
                    f"p={result['fraud_probability']:.3f}  "
                    f"latency={result['latency_ms']:.1f}ms"
                )

            if counts["processed"] % HEARTBEAT_N == 0:
                n = counts["processed"]
                prec = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
                reca = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
                log.info(
                    f"📊  Processed={n:,}  Fraud={counts['fraud']}  "
                    f"TP={counts['tp']} TN={counts['tn']} "
                    f"FP={counts['fp']} FN={counts['fn']}  "
                    f"Precision={prec:.2%}  Recall={reca:.2%}"
                )

        except requests.Timeout:
            counts["timeouts"] += 1
            log.error(f"⏱️  TIMEOUT txn={txn_id[:14]} (>{API_TIMEOUT_S*1000:.0f}ms)")
        except Exception as exc:
            counts["errors"] += 1
            log.error(f"❌  Error txn={txn_id[:14]}: {exc}")


if __name__ == "__main__":
    run()
