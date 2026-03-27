"""
Generate Test Sample CSV for Fraud Detection Testing

This script creates synthetic credit card transaction data for testing
the FraudShield AI system with customizable sample size and fraud ratio.

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


def generate_test_data(n_samples: int, fraud_ratio: float, seed: int = 42):
    """
    Generate synthetic credit card transaction data.

    Args:
        n_samples: Total number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        pandas DataFrame with synthetic transaction data
    """
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Feature names (V1 to V28 - PCA components from original dataset)
    feature_names = [f"V{i}" for i in range(1, 29)]

    # Generate legitimate transactions (tighter distribution around zero)
    legit_features = np.random.randn(n_legit, 28) * 0.5
    legit_time = np.random.randint(0, 100000, n_legit)
    legit_amount = np.abs(np.random.exponential(50, n_legit))
    legit_class = np.zeros(n_legit)

    # Generate fraudulent transactions (wider distribution - more extreme values)
    fraud_features = np.random.randn(n_fraud, 28) * 2.5
    fraud_time = np.random.randint(0, 100000, n_fraud)
    fraud_amount = np.abs(np.random.exponential(500, n_fraud))
    fraud_class = np.ones(n_fraud)

    # Combine datasets
    features = np.vstack([legit_features, fraud_features])
    times = np.concatenate([legit_time, fraud_time])
    amounts = np.concatenate([legit_amount, fraud_amount])
    classes = np.concatenate([legit_class, fraud_class])

    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"] = classes.astype(int)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic credit card test data for fraud detection testing",
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

    # Validate inputs
    if args.samples < 1:
        parser.error("Number of samples must be at least 1")
    if not 0 <= args.fraud_ratio <= 1:
        parser.error("Fraud ratio must be between 0.0 and 1.0")

    n_fraud = int(args.samples * args.fraud_ratio)
    n_legit = args.samples - n_fraud

    print(f"Generating {args.samples} samples ({n_fraud} fraud, {n_legit} legitimate)...")

    # Generate data
    df = generate_test_data(args.samples, args.fraud_ratio, args.seed)

    # Determine output filename
    if args.output:
        output_path = args.output
    else:
        fraud_pct = int(args.fraud_ratio * 100)
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"test_sample_{args.samples}_{fraud_pct}pct.csv"
        )

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print summary
    actual_fraud = df["Class"].sum()
    actual_ratio = actual_fraud / len(df)

    print(f"\nGenerated test dataset saved to: {output_path}")
    print(f"\nSummary:")
    print(f"   Total transactions: {len(df)}")
    print(f"   Legitimate: {len(df) - int(actual_fraud)}")
    print(f"   Fraudulent: {int(actual_fraud)}")
    print(f"   Fraud ratio: {actual_ratio:.2%}")
    print(f"   Columns: {len(df.columns)} (Time, V1-V28, Amount, Class)")


if __name__ == "__main__":
    main()
