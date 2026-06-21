import pandas as pd
import numpy as np
import os

print("="*70)
print("FEATURE ENGINEERING FOR LSTM - MULTI-STOCK (BULLETPROOF)")
print("="*70)

# Load combined data
df = pd.read_csv('data/multi_stock/combined_raw.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\nLoaded {len(df)} rows from {df['ticker'].nunique()} stocks")

# Check columns
print(f"Columns: {list(df.columns)}")

# ============================================================================
# FEATURE ENGINEERING (PER STOCK)
# ============================================================================

def engineer_features(stock_df):
    df = stock_df.sort_values('date').reset_index(drop=True)

    # Lag features
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag5'] = df['close'].shift(5)

    # Moving averages
    df['sma_5']  = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()

    # Daily returns
    df['daily_return'] = df['close'].pct_change() * 100

    # RSI — FIX 1: guard against avg_loss == 0
    delta    = df['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']           = ema_12 - ema_26
    df['macd_signal']    = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands — FIX 2: reuse sma_20, no redundant recompute
    std_20          = df['close'].rolling(window=20).std()
    df['bb_upper']  = df['sma_20'] + (std_20 * 2)
    df['bb_middle'] = df['sma_20']
    df['bb_lower']  = df['sma_20'] - (std_20 * 2)

    # Target
    df['target_close'] = df['close'].shift(-1)

    # FIX 3: drop NaNs including target_close — no more fragile iloc[:-1]
    df = df.dropna(subset=[
        'sma_20', 'rsi_14', 'macd',
        'bb_upper', 'close_lag1', 'target_close'
    ])
    return df
        

# ============================================================================
# APPLY TO EACH STOCK
# ============================================================================

engineered_dfs = []
for ticker in sorted(df['ticker'].unique()):
    print(f"\nProcessing {ticker}...")
    stock_data = df[df['ticker'] == ticker].copy()
    stock_data = engineer_features(stock_data)
    
    if len(stock_data) > 0:
        engineered_dfs.append(stock_data)
        print(f"  ✓ {len(stock_data)} valid rows")
    else:
        print(f"  ✗ No valid rows!")

# ============================================================================
# COMBINE AND SAVE
# ============================================================================

if engineered_dfs:
    df = pd.concat(engineered_dfs, ignore_index=True)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/lstm_features.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Features saved: data/processed/lstm_features.csv")
    print(f"✓ Total LSTM samples: {len(df)}")
    print(f"\nRows per ticker:")
    print(df['ticker'].value_counts().sort_index())
    print(f"\n✓ Ready for preprocessing!")
    print("="*70)
else:
    print("\n✗ ERROR: No valid data!")