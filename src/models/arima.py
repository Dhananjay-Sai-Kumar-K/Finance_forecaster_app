from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def fit_predict_arima(series, order=(5,1,0), steps=1):
    """
    Fits an ARIMA model and forecasts future steps.
    
    Args:
        series (pd.Series): Time series data.
        order (tuple): ARIMA order (p, d, q).
        steps (int): Number of steps to forecast.
        
    Returns:
        forecast (pd.Series): Forecasted values.
    """
    try:
        # Ensure series is numeric and drop NaNs
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return pd.Series([series.iloc[-1]] * steps) # Fallback to last value
