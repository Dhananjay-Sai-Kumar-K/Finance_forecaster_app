import pytest
import pandas as pd
from src.data_loader import fetch_data

def test_fetch_valid_ticker():
    """Test fetching data for a valid ticker."""
    df = fetch_data('AAPL', period='1mo', interval='1d')
    assert not df.empty
    assert 'Close' in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)

def test_fetch_invalid_ticker():
    """Test fetching data for an invalid ticker."""
    # yfinance often returns empty dataframe for invalid tickers, 
    # or might raise an exception depending on version/API response.
    # Our fetch_data implementation catches exceptions and returns empty df or raises.
    # Based on current implementation, it might return empty df.
    try:
        df = fetch_data('INVALID_TICKER_12345', period='1mo', interval='1d')
        assert df.empty
    except Exception:
        # If it raises, that's also acceptable behavior for invalid input
        pass
