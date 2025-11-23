import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD

def add_technical_indicators(df):
    """
    Adds technical indicators (RSI, MACD) to the DataFrame.
    """
    df = df.copy()
    # Ensure Close is 1D
    close_prices = df['Close']
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    # RSI
    rsi = RSIIndicator(close=close_prices, window=14)
    df['RSI'] = rsi.rsi()

    # MACD
    macd = MACD(close=close_prices)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    df.dropna(inplace=True)
    return df

def normalize_data(data):
    """
    Normalizes data using MinMaxScaler.
    If data is a DataFrame, normalizes all columns.
    Returns the scaler and normalized data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

def prepare_lstm_data(data, lookback=60, target_col_index=0):
    """
    Creates sequences for LSTM training.
    
    Args:
        data (np.array): Normalized data (samples, features).
        lookback (int): Number of previous time steps to use for prediction.
        target_col_index (int): Index of the target column (usually 'Close' is 0).
        
    Returns:
        tuple: (X, y)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i]) # All features
        y.append(data[i, target_col_index]) # Target only
    
    return np.array(X), np.array(y)
