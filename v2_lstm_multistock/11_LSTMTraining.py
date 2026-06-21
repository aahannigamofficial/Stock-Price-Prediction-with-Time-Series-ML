import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

print("="*70)
print("LSTM MODEL TRAINING")
print("="*70)

# Load preprocessed data
print("\nLoading data...")
X_train = joblib.load('data/X_train_lstm.pkl')
y_train = joblib.load('data/y_train_lstm.pkl')
X_val = joblib.load('data/X_val_lstm.pkl')
y_val = joblib.load('data/y_val_lstm.pkl')
X_test = joblib.load('data/X_test_lstm.pkl')
y_test = joblib.load('data/y_test_lstm.pkl')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================================
# BUILD LSTM MODEL
# ============================================================================

print("\nBuilding LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print(model.summary())

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\nTraining LSTM model (this may take 5-10 minutes)...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

y_pred = model.predict(X_test, verbose=0)
y_pred = y_pred.flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\nTest Set Metrics:")
print(f"  MAE: ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  R² Score: {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

directional_accuracy = np.mean((y_pred[1:] > y_test[:-1]) == (y_test[1:] > y_test[:-1])) * 100
print(f"  Directional Accuracy: {directional_accuracy:.2f}%")

# ============================================================================
# SAVE MODEL
# ============================================================================

os.makedirs('models', exist_ok=True)

model.save('models/lstm_model.h5')
print(f"\n✓ Model saved: models/lstm_model.h5")

# Save metrics
metrics = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'mape': mape,
    'directional_accuracy': directional_accuracy
}
joblib.dump(metrics, 'models/lstm_metrics.pkl')
print(f"✓ Metrics saved: models/lstm_metrics.pkl")

print("\n" + "="*70)
print("LSTM training complete! Ready to deploy.")
print("="*70)