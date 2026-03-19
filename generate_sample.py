import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)

# Generate parameters
n_samples = 1000
fraud_ratio = 0.05
n_fraud = int(n_samples * fraud_ratio)
n_legit = n_samples - n_fraud

print(f"Generating {n_samples} samples ({n_fraud} fraud, {n_legit} legitimate)...")

# Feature names (V1 to V28)
feature_names = [f"V{i}" for i in range(1, 29)]

# Generate legitimate transactions (normal distribution)
legit_features = np.random.randn(n_legit, 28) * 0.5
legit_time = np.random.randint(0, 100000, n_legit)
legit_amount = np.abs(np.random.exponential(50, n_legit))
legit_class = np.zeros(n_legit)

# Generate fraudulent transactions (more extreme values)
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
df["Class"] = classes

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), "sample_creditcard.csv")
df.to_csv(output_path, index=False)

print(f"✅ Generated sample dataset saved to: {output_path}")
