import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

print("="*70)
print("LSTM PREPROCESSING - SCALING & SEQUENCES")
print("="*70)

# Load features
df = pd.read_csv('data/processed/lstm_features.csv')
print(f"\nLoaded {len(df)} rows")

# ============================================================================
# SEPARATE FEATURES AND TARGET
# ============================================================================

# Features to use (exclude these columns)
exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'target_close']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['target_close'].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# ============================================================================
# SCALE FEATURES
# ============================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures scaled (mean=0, std=1)")

# ============================================================================
# CREATE SEQUENCES FOR LSTM
# ============================================================================

def create_sequences(X, y, lookback=30):
    """Create sequences for LSTM"""
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    
    return np.array(X_seq), np.array(y_seq)

lookback = 30
X_seq, y_seq = create_sequences(X_scaled, y, lookback=lookback)

print(f"\nSequences created (lookback={lookback})")
print(f"X_seq shape: {X_seq.shape}")
print(f"y_seq shape: {y_seq.shape}")

# ============================================================================
# TRAIN-TEST SPLIT (CHRONOLOGICAL)
# ============================================================================

train_size = int(len(X_seq) * 0.70)
val_size = int(len(X_seq) * 0.15)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]

X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"\nTrain: {len(X_train)} samples")
print(f"Val: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================

os.makedirs('data', exist_ok=True)

joblib.dump(X_train, 'data/X_train_lstm.pkl')
joblib.dump(y_train, 'data/y_train_lstm.pkl')
joblib.dump(X_val, 'data/X_val_lstm.pkl')
joblib.dump(y_val, 'data/y_val_lstm.pkl')
joblib.dump(X_test, 'data/X_test_lstm.pkl')
joblib.dump(y_test, 'data/y_test_lstm.pkl')
joblib.dump(scaler, 'data/scaler_lstm.pkl')

print(f"\n✓ All data saved!")
print("="*70)
print("Ready for LSTM training!")
print("="*70)