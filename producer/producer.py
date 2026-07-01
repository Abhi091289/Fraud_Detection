"""
producer/producer.py
Streams real Credit Card Fraud dataset rows to Kafka.

Instead of fake data, this replays actual transactions from the
real dataset — preserving original feature values and true labels.
Rate: ~10 transactions/second (adjustable via RATE env var).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
import uuid

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DATA_PATH     = os.getenv("DATA_PATH", "/data/creditcard.csv")
TOPIC         = "transactions.raw"
RATE          = float(os.getenv("RATE", "10"))   # transactions per second
LOOP          = os.getenv("LOOP", "true").lower() == "true"  # replay indefinitely
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


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


def load_dataset(path: str) -> list[dict]:
    """Load CSV into list of dicts — keeps all columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run:  python data/download_dataset.py  first"
        )
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_message(row: dict) -> dict:
    """Convert CSV row to Kafka message payload."""
    return {
        "transaction_id":  str(uuid.uuid4()),
        "Amount":          float(row.get("Amount", 0)),
        "Time":            float(row.get("Time", 0)),
        "true_label":      int(float(row.get("Class", 0))),  # 0=legit, 1=fraud
        **{f"V{i}": float(row.get(f"V{i}", 0)) for i in range(1, 29)},
    }


def run() -> None:
    wait_for_kafka()

    log.info(f"Loading dataset from {DATA_PATH} ...")
    rows = load_dataset(DATA_PATH)
    n_fraud = sum(1 for r in rows if int(float(r.get("Class", 0))) == 1)
    log.info(
        f"  Loaded {len(rows):,} rows | "
        f"fraud: {n_fraud} ({n_fraud/len(rows):.3%}) | "
        f"loop: {LOOP} | rate: {RATE}/sec"
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=5,
    )

    interval  = 1.0 / RATE
    sent      = 0
    fraud_sent = 0
    loop_num  = 0

    log.info(f"🚀  Streaming to '{TOPIC}' at {RATE}/sec ...")

    try:
        while True:
            loop_num += 1
            # Shuffle on each loop so fraud isn't always at same positions
            loop_rows = rows.copy()
            random.shuffle(loop_rows)

            for row in loop_rows:
                msg = build_message(row)
                producer.send(
                    topic=TOPIC,
                    key=msg["transaction_id"],
                    value=msg,
                )
                sent += 1
                if msg["true_label"] == 1:
                    fraud_sent += 1

                if sent % 500 == 0:
                    log.info(
                        f"📨  Sent {sent:,} txns | "
                        f"fraud in stream: {fraud_sent} ({fraud_sent/sent:.2%}) | "
                        f"loop #{loop_num}"
                    )

                time.sleep(interval)

            if not LOOP:
                log.info(f"✅  Dataset exhausted after {sent:,} messages. Stopping.")
                break

    except KeyboardInterrupt:
        log.info("Stopping producer ...")
    finally:
        producer.flush()
        producer.close()
        log.info(f"Total sent: {sent:,}")


if __name__ == "__main__":
    run()
