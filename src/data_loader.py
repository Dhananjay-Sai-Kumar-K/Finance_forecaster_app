import yfinance as yf
import pandas as pd

def fetch_data(ticker, period='1y', interval='1d'):
    """
    Fetches historical market data for a given ticker.
    
    Args:
        ticker (str): The stock/crypto ticker (e.g., 'AAPL', 'BTC-USD').
        period (str): Data period to download (e.g., '1d', '1mo', '1y', 'max').
        interval (str): Data interval (e.g., '1m', '1h', '1d').
        
    Returns:
        pd.DataFrame: DataFrame containing the historical data.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # Check which level has 'Close'
            if 'Close' in data.columns.get_level_values(0):
                data.columns = data.columns.get_level_values(0)
            elif 'Close' in data.columns.get_level_values(1):
                data.columns = data.columns.get_level_values(1)
            
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_latest_price(ticker):
    """
    Fetches the latest available price for a given ticker.
    
    Args:
        ticker (str): The stock/crypto ticker.
        
    Returns:
        float: The latest price.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        # Try fast info first
        if hasattr(ticker_obj, 'fast_info') and 'last_price' in ticker_obj.fast_info:
             return ticker_obj.fast_info['last_price']
        
        # Fallback to history
        data = ticker_obj.history(period='1d', interval='1m')
        if not data.empty:
            # Flatten if needed
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    data.columns = data.columns.get_level_values(0)
                elif 'Close' in data.columns.get_level_values(1):
                    data.columns = data.columns.get_level_values(1)
                    
            return data['Close'].iloc[-1]
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching latest price for {ticker}: {e}")
        return 0.0
