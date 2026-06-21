import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# ============================================================================
# CONFIGURATION
# ============================================================================

TICKER = "AAPL"
END_DATE = datetime.now().date()  
START_DATE = END_DATE - timedelta(days=3*365)

# ============================================================================
# DOWNLOAD AND CLEAN DATA
# ============================================================================

print(f"Downloading {TICKER} from {START_DATE} to {END_DATE}...")

# Download stock data
df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

print(f"✓ Downloaded {len(df)} rows")

# Flatten MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Reset index so Date becomes a column
df = df.reset_index()

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Create data folder if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save raw data
output_file = 'data/raw/aapl_raw.csv'
df.to_csv(output_file, index=False)
print(f"✓ Data saved to {output_file}")

#
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())