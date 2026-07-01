"""
data/download_dataset.py

Downloads the Credit Card Fraud Detection dataset.
Source: OpenML dataset #1597 (same as Kaggle mlg-ulb/creditcardfraud)
No credentials required — completely free.

Dataset stats:
  - 284,807 transactions
  - 492 fraudulent (0.172%)
  - 28 PCA features (V1-V28) + Amount + Time

Usage:
    python data/download_dataset.py
"""

import os
import sys
import time

OUT_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")


def download_from_openml() -> bool:
    """Download via scikit-learn's OpenML connector (no credentials needed)."""
    print("📥  Downloading Credit Card Fraud dataset from OpenML...")
    print("    This downloads ~143 MB and caches locally.")
    print("    First download takes 2–5 minutes. Subsequent runs are instant.\n")

    try:
        from sklearn.datasets import fetch_openml
        import pandas as pd

        t0 = time.time()
        dataset = fetch_openml(
            data_id=1597,
            as_frame=True,
            parser="auto",
            cache=True,
        )
        elapsed = time.time() - t0

        df = dataset.frame

        # Rename target column if needed
        if "Class" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "Class"})

        # Ensure Class is integer
        df["Class"] = df["Class"].astype(str).str.strip().astype(float).astype(int)

        # Ensure V1-V28, Amount, Time are float
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Save
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        df.to_csv(OUT_PATH, index=False)

        print(f"✅  Dataset saved → {OUT_PATH}")
        print(f"    Rows        : {len(df):,}")
        print(f"    Fraud (1)   : {(df['Class'] == 1).sum():,}  ({(df['Class'] == 1).mean():.3%})")
        print(f"    Legit (0)   : {(df['Class'] == 0).sum():,}")
        print(f"    Download    : {elapsed:.0f}s")
        return True

    except Exception as e:
        print(f"⚠️   OpenML download failed: {e}")
        return False


def generate_synthetic_fallback() -> None:
    """
    Generate a synthetic dataset that closely mirrors the real Credit Card Fraud
    dataset's statistical properties. Used only if OpenML is unavailable.
    """
    print("\n🔄  Generating synthetic fallback dataset...")
    print("    This mirrors the real dataset's distributions.\n")

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_legit  = 284315
    n_fraud  = 492

    def make_legit(n):
        data = {"Class": np.zeros(n, dtype=int)}
        for i in range(1, 29):
            data[f"V{i}"] = rng.normal(0, 1, n)
        data["Amount"] = np.round(rng.lognormal(3.5, 1.2, n), 2)
        data["Time"]   = np.sort(rng.uniform(0, 172792, n))
        return pd.DataFrame(data)

    def make_fraud(n):
        data = {"Class": np.ones(n, dtype=int)}
        # Fraud has distinctive patterns in V1, V3, V4, V10, V11, V12, V14, V16
        vecs = rng.normal(0, 1, (n, 28))
        vecs[:, 0]  = rng.normal(-4.77, 4.5,  n)   # V1
        vecs[:, 2]  = rng.normal(-7.03, 6.7,  n)   # V3
        vecs[:, 3]  = rng.normal( 4.54, 3.0,  n)   # V4
        vecs[:, 9]  = rng.normal(-4.53, 4.2,  n)   # V10
        vecs[:, 11] = rng.normal(-5.72, 5.1,  n)   # V12
        vecs[:, 13] = rng.normal(-7.07, 4.1,  n)   # V14
        for i in range(28):
            data[f"V{i+1}"] = vecs[:, i]
        data["Amount"] = np.round(rng.lognormal(4.5, 1.5, n), 2)
        data["Time"]   = rng.uniform(0, 172792, n)
        return pd.DataFrame(data)

    df = (
        pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"✅  Synthetic dataset saved → {OUT_PATH}")
    print(f"    Rows        : {len(df):,}")
    print(f"    Fraud rate  : {df['Class'].mean():.3%}")
    print(f"\n    Note: For best results, use the real dataset.")
    print(f"    Install kaggle CLI and run:")
    print(f"    kaggle datasets download mlg-ulb/creditcardfraud")


if __name__ == "__main__":
    if os.path.exists(OUT_PATH):
        import pandas as pd
        df = pd.read_csv(OUT_PATH, nrows=5)
        print(f"✅  Dataset already exists at {OUT_PATH}")
        print(f"    To re-download, delete the file and run again.")
        sys.exit(0)

    success = download_from_openml()
    if not success:
        generate_synthetic_fallback()

    print(f"\n👉  Next step: python training/train.py")
