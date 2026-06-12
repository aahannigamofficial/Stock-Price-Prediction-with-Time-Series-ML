import sys
print("DEBUG: Script is running", file=sys.stderr)

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

print("Script started!")

TICKER = "AAPL"
END_DATE = datetime.now().date()  
START_DATE = END_DATE - timedelta(days=3*365)  

def download_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✓ Downloaded {len(data)} rows of data for {ticker}")
        return data
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return None

if __name__ == "__main__":
    # Create data directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("="*70)
    print(f"Downloading {TICKER} data from {START_DATE} to {END_DATE}...")
    print("="*70)
    
    # Download data
    df = download_stock_data(TICKER, START_DATE, END_DATE)

    print(f"DEBUG: df is not None = {df is not None}")
    print(f"DEBUG: df type = {type(df)}")
    
    if df is not None:
        print("DEBUG: Inside if block")
        print(f"DEBUG: Is MultiIndex? {isinstance(df.columns, pd.MultiIndex)}")
        
        # Flatten column names if they're MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            print("DEBUG: Flattening MultiIndex...")
            df.columns = [col[0] for col in df.columns.values]
            print(f"DEBUG: After flatten - columns = {df.columns}")
        
        # Make sure Date is a proper column (not index)
        print("DEBUG: Resetting index...")
        df = df.reset_index()
        
        # Rename columns to lowercase for consistency
        print("DEBUG: Converting to lowercase...")
        df.columns = df.columns.str.lower()
        print(f"DEBUG: Final columns = {list(df.columns)}")
        
        # Save raw data
        output_file = f'data/raw/{TICKER}_raw.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Data saved to {output_file}")
        
        # Print basic info
        print(f"\nDATA INFO")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
    else:
        print("Failed to download data")