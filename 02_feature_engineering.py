import yfinance as yf
import numpy as np
import pandas as pd

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Load raw data
input_file = 'data/raw/aapl_raw.csv'
df = pd.read_csv(input_file)

#Calculate Simple moving average,Daily Return and Lag feature(Previous Day's close)
SMA_5 = df['close'].rolling(window=5).mean()
daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
close_lag1 = df['close'].shift(1)

df['sma_5'] = SMA_5
df['daily_return'] = daily_return
df['close_lag1'] = close_lag1

df.to_csv('data/processed/aapl_features.csv', index=False)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData saved to data/processed/aapl_features.csv")