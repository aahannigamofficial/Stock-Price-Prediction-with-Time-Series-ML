import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load features
df = pd.read_csv('data/processed/aapl_features.csv')

# Create target variable (next day's closing price)
df['target_close'] = df['close'].shift(-1)

# Drop NaN rows
df = df.dropna()

print(f"Rows after dropping NaN: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLast few rows:")
print(df.tail())

# Separate features (X) and target (y)
X = df.drop(['date', 'target_close'], axis=1)  # Drop non-features
y = df['target_close']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\nTrain: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaling complete!")
print(f"X_train_scaled shape: {X_train_scaled.shape}")