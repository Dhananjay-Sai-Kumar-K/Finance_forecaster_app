from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

def fit_predict_arima(series, order=(5,1,0), steps=1):
    """
    Fits an ARIMA model and forecasts future steps.
    
    Args:
        series (pd.Series): Time series data.
        order (tuple): ARIMA order (p, d, q).
        steps (int): Number of steps to forecast.
        
    Returns:
        tuple: (forecast, metrics_dict) where metrics_dict contains MAE and RMSE
    """
    try:
        # Ensure series is numeric and drop NaNs
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        if series.empty:
            raise ValueError("Time series data is empty after preprocessing.")
        
        # Split data for validation (80/20)
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        # Fit model on training data
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        # Make predictions on test set for metrics
        predictions = model_fit.forecast(steps=len(test))
        
        # Calculate metrics
        # Calculate metrics (use .values to avoid index mismatch)
        mae = np.mean(np.abs(predictions.values - test.values))
        rmse = np.sqrt(np.mean((predictions.values - test.values) ** 2))
        
        # Now fit on full data for final forecast
        model_full = ARIMA(series, order=order)
        model_full_fit = model_full.fit()
        forecast = model_full_fit.forecast(steps=steps)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'Order': order
        }
        
        return forecast, metrics
    except Exception as e:
        print(f"ARIMA Error: {e}")
        # Fallback to last value
        if not series.empty:
            last_val = series.iloc[-1]
        else:
            last_val = 0.0
            
        fallback_forecast = pd.Series([last_val] * steps)
        fallback_metrics = {'MAE': 0, 'RMSE': 0, 'Order': order}
        return fallback_forecast, fallback_metrics
