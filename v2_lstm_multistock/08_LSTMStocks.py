import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "META", "NVDA"]
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=3*365)

print("="*70)
print("DOWNLOADING MULTIPLE STOCKS FOR LSTM TRAINING")
print("="*70)
print(f"\nDate range: {START_DATE} to {END_DATE}")
print(f"Tickers: {TICKERS}\n")

# ============================================================================
# DOWNLOAD ALL STOCKS
# ============================================================================

all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)

        if df.empty:
            print(f"  ✗ No data returned for {ticker}")
            continue

        # yfinance returns MultiIndex columns like ('Close', 'AAPL')
        # Drop the ticker level, keep just the column name
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Lowercase column names
        df.columns = df.columns.str.lower()

        # The date is in the index — pull it out explicitly
        df.index.name = 'date'
        df = df.reset_index()

        # Lowercase again in case reset_index brought back 'Date'
        df.columns = df.columns.str.lower()

        df['ticker'] = ticker
        all_data.append(df)
        print(f"  ✓ {ticker}: {len(df)} rows")

    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {e}")

# ============================================================================
# COMBINE AND SAVE
# ============================================================================

if not all_data:
    print("\n✗ No data downloaded successfully. Exiting.")
    exit(1)

combined_df = pd.concat(all_data, ignore_index=True)

# Sanity check columns
print(f"\nColumns after concat: {list(combined_df.columns)}")

expected_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'ticker'}
missing = expected_cols - set(combined_df.columns)
if missing:
    print(f"\n✗ Missing expected columns: {missing}")
    exit(1)

# Keep only what we need
combined_df = combined_df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

# Sort by ticker then date
combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)

# Save
os.makedirs('data/multi_stock', exist_ok=True)
combined_df.to_csv('data/multi_stock/combined_raw.csv', index=False)

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print(f"COMBINED DATA SUMMARY")
print(f"{'='*70}")
print(f"Total rows    : {len(combined_df)}")
print(f"Tickers       : {combined_df['ticker'].unique().tolist()}")
print(f"Date range    : {combined_df['date'].min()} to {combined_df['date'].max()}")
print(f"Columns       : {list(combined_df.columns)}")
print(f"\nRows per ticker:")
print(combined_df['ticker'].value_counts().sort_index().to_string())
print(f"\n✓ Saved: data/multi_stock/combined_raw.csv")
print("="*70)