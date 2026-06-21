import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from statsmodels.tsa.arima.model import ARIMA
def create_sequences(X, y, lookback=30):
    X_seq = []
    y_seq = []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

X_train_scaled = joblib.load('data/X_train_scaled.pkl')
y_train = joblib.load('data/y_train.pkl')
X_test_scaled = joblib.load('data/X_test_scaled.pkl')
y_test = joblib.load('data/y_test.pkl')

model1 = LinearRegression()
model1.fit(X_train_scaled,y_train)
y_pred1 = model1.predict(X_test_scaled)
mae = np.mean(np.abs(y_pred1 - y_test))
print(f"Linear Regression MAE: {mae}")

model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)
mae2 = np.mean(np.abs(y_pred2 - y_test))
print(f"Random Forest MAE: {mae2}")

model3 = XGBRegressor()
model3.fit(X_train_scaled,y_train)
y_pred3 = model3.predict(X_test_scaled)
mae3 = np.mean(np.abs(y_pred3 - y_test))
print(f"XGBoost MAE: {mae3:.4f}")

y_train_np = y_train.values
y_test_np = y_test.values
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_np, lookback=30)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_np, lookback=30)

model4 = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, 15)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model4.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model4.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=1)
y_pred4 = model4.predict(X_test_seq)
mae4 = np.mean(np.abs(y_pred4 - y_test_seq))
print(f"LSTM MAE: {mae4:.4f}")

model5 = ARIMA(y_train_np, order=(5, 1, 2))
model5_fit = model5.fit()
y_pred5 = model5_fit.forecast(steps=len(y_test_np))
mae5 = np.mean(np.abs(y_pred5 - y_test_np))
print(f"ARIMA MAE: {mae5:.4f}")

joblib.dump(model1, 'data/model1.pkl')
print("✓ Linear Regression model saved!")