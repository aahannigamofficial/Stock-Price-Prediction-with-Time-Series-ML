import pandas as pd

# =============================================================================
# FUNCTIONS
# =============================================================================


def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    SMA_5 = prices.rolling(window=window).mean()
    return SMA_5


def calculate_daily_return(prices):
    """Calculate Daily Returns"""
    daily_return = (prices - prices.shift(1)) / prices.shift(1) * 100
    return daily_return


def calculate_lag_features(prices, lag=1):
    """Calculate Lag Features"""
    close_lag1 = prices.shift(lag)
    return close_lag1


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0)
    loss = -delta.where(delta < 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Load raw data


input_file = 'data/raw/aapl_raw.csv'

df = pd.read_csv(input_file)

df['sma_5'] = calculate_sma(df['close'], window=5)
df['daily_return'] = calculate_daily_return(df['close'])
df['close_lag1'] = calculate_lag_features(df['close'], lag=1)
df['rsi_14'] = calculate_rsi(df['close'], period=14)
df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df['close'])
df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])

df.to_csv('data/processed/aapl_features.csv', index=False)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
print("\nData saved to data/processed/aapl_features.csv")
