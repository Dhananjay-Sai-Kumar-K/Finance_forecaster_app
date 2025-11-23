from src.data_loader import fetch_data
import pandas as pd

try:
    print("Fetching data for AAPL...")
    df = fetch_data('AAPL', period='1mo', interval='1d')
    print("Columns:", df.columns)
    if isinstance(df.columns, pd.MultiIndex):
        print("Levels:", df.columns.levels)
    
    print("Checking 'Close' column:")
    close = df['Close']
    print("Type of df['Close']:", type(close))

except Exception as e:
    print("Top level error:", e)
