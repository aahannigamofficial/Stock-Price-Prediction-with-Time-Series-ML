import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

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