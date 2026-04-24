"""
Generate a sample dataset by stratified sampling from the real creditcard.csv.
Falls back to synthetic generation if the real dataset is unavailable.
"""
import numpy as np
import pandas as pd
import os

# Parameters
n_samples = 1000
fraud_ratio = 0.05
n_fraud = int(n_samples * fraud_ratio)
n_legit = n_samples - n_fraud
seed = 42

base_dir = os.path.dirname(__file__)
real_csv = os.path.join(base_dir, "creditcard.csv")
output_path = os.path.join(base_dir, "sample_creditcard.csv")

if os.path.exists(real_csv):
    print(f"Found real dataset: {real_csv}")
    print(f"Stratified sampling {n_legit} legit + {n_fraud} fraud...")

    df_full = pd.read_csv(real_csv)
    df_legit = df_full[df_full["Class"] == 0]
    df_fraud = df_full[df_full["Class"] == 1]

    # Sample with replacement if not enough fraud rows
    sampled_legit = df_legit.sample(n=n_legit, random_state=seed, replace=len(df_legit) < n_legit)
    sampled_fraud = df_fraud.sample(n=n_fraud, random_state=seed, replace=len(df_fraud) < n_fraud)

    df = pd.concat([sampled_legit, sampled_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"[OK] Stratified sample saved to: {output_path}")
    print(f"     {len(df)} rows, {int(df['Class'].sum())} fraud ({df['Class'].mean():.2%})")

else:
    print(f"[WARN] Real dataset not found at {real_csv}")
    print("Falling back to synthetic generation...")

    np.random.seed(seed)
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
    df["Class"] = classes

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"[OK] Synthetic sample saved to: {output_path}")
    print(f"     {len(df)} rows, {int(df['Class'].sum())} fraud")
