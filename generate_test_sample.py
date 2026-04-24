"""
Generate Test Sample CSV for Fraud Detection Testing

Creates a test dataset by stratified sampling from the real creditcard.csv.
Falls back to synthetic generation if the real dataset is unavailable.

Usage:
    python generate_test_sample.py                           # 200 samples, 3% fraud
    python generate_test_sample.py --samples 100             # 100 samples, 3% fraud
    python generate_test_sample.py --fraud-ratio 0.05        # 200 samples, 5% fraud
    python generate_test_sample.py --samples 500 --fraud-ratio 0.02  # 500 samples, 2% fraud
"""

import argparse
import numpy as np
import pandas as pd
import os


def generate_from_real_data(n_samples: int, fraud_ratio: float, seed: int = 42):
    """
    Generate test data by stratified sampling from real creditcard.csv.

    Returns:
        pandas DataFrame, or None if real data is unavailable
    """
    base_dir = os.path.dirname(__file__)
    real_csv = os.path.join(base_dir, "creditcard.csv")

    if not os.path.exists(real_csv):
        return None

    df_full = pd.read_csv(real_csv)
    df_legit = df_full[df_full["Class"] == 0]
    df_fraud = df_full[df_full["Class"] == 1]

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    sampled_legit = df_legit.sample(n=n_legit, random_state=seed, replace=len(df_legit) < n_legit)
    sampled_fraud = df_fraud.sample(n=n_fraud, random_state=seed, replace=len(df_fraud) < n_fraud)

    df = pd.concat([sampled_legit, sampled_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_synthetic_data(n_samples: int, fraud_ratio: float, seed: int = 42):
    """
    Fallback: Generate synthetic credit card transaction data.

    Returns:
        pandas DataFrame
    """
    np.random.seed(seed)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    feature_names = [f"V{i}" for i in range(1, 29)]

    legit_features = np.random.randn(n_legit, 28) * 0.5
    legit_time = np.random.randint(0, 100000, n_legit)
    legit_amount = np.abs(np.random.exponential(50, n_legit))
    legit_class = np.zeros(n_legit)

    fraud_features = np.random.randn(n_fraud, 28) * 2.5
    fraud_time = np.random.randint(0, 100000, n_fraud)
    fraud_amount = np.abs(np.random.exponential(500, n_fraud))
    fraud_class = np.ones(n_fraud)

    features = np.vstack([legit_features, fraud_features])
    times = np.concatenate([legit_time, fraud_time])
    amounts = np.concatenate([legit_amount, fraud_amount])
    classes = np.concatenate([legit_class, fraud_class])

    df = pd.DataFrame(features, columns=feature_names)
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"] = classes.astype(int)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for fraud detection testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_test_sample.py                            # 200 samples, 3% fraud
    python generate_test_sample.py --samples 100              # 100 samples, 3% fraud
    python generate_test_sample.py --fraud-ratio 0.05         # 200 samples, 5% fraud
    python generate_test_sample.py --samples 500 --fraud-ratio 0.02  # 500 samples, 2% fraud
        """
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=200,
        help="Number of transactions to generate (default: 200)"
    )
    parser.add_argument(
        "--fraud-ratio", "-f",
        type=float,
        default=0.03,
        help="Fraud ratio between 0.0 and 1.0 (default: 0.03 = 3%%)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename (default: auto-generated based on parameters)"
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("Number of samples must be at least 1")
    if not 0 <= args.fraud_ratio <= 1:
        parser.error("Fraud ratio must be between 0.0 and 1.0")

    n_fraud = max(1, int(args.samples * args.fraud_ratio))
    n_legit = args.samples - n_fraud

    print(f"Generating {args.samples} samples ({n_fraud} fraud, {n_legit} legitimate)...")

    # Try real data first, fallback to synthetic
    df = generate_from_real_data(args.samples, args.fraud_ratio, args.seed)
    if df is not None:
        source = "stratified sample from creditcard.csv"
    else:
        print("[WARN] Real dataset not found, falling back to synthetic generation.")
        df = generate_synthetic_data(args.samples, args.fraud_ratio, args.seed)
        source = "synthetic generation"

    # Determine output filename
    if args.output:
        output_path = args.output
    else:
        fraud_pct = int(args.fraud_ratio * 100)
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"test_sample_{args.samples}_{fraud_pct}pct.csv"
        )

    df.to_csv(output_path, index=False)

    actual_fraud = df["Class"].sum()
    actual_ratio = actual_fraud / len(df)

    print(f"\nGenerated test dataset saved to: {output_path}")
    print(f"  Source: {source}")
    print(f"  Total transactions: {len(df)}")
    print(f"  Legitimate: {len(df) - int(actual_fraud)}")
    print(f"  Fraudulent: {int(actual_fraud)}")
    print(f"  Fraud ratio: {actual_ratio:.2%}")
    print(f"  Columns: {len(df.columns)} ({', '.join(df.columns[:3])}...{df.columns[-1]})")


if __name__ == "__main__":
    main()
