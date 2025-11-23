import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from src.data_loader import fetch_data, get_latest_price
from src.preprocessing import add_technical_indicators, normalize_data, prepare_lstm_data
from src.models.arima import fit_predict_arima
from src.models.lstm import build_lstm_model, train_lstm_model, predict_lstm
# from src.models.rl_agent import train_rl_agent, StockTradingEnv # Importing this might be heavy for live loop, load on demand

st.set_page_config(page_title="Finance Forecaster", layout="wide")

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("src/style.css")

st.title("Finance Forecasting System 📈")

# Sidebar
st.sidebar.header("Configuration")

# Asset Selection
asset_class = st.sidebar.selectbox("Asset Class", ["Stocks", "Crypto", "Commodities", "Indices", "Forex", "Custom"])

POPULAR_TICKERS = {
    "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "ADA-USD"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"], # Gold, Silver, Oil, Gas, Copper
    "Indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE"], # S&P 500, Dow, Nasdaq, Russell 2000, FTSE 100
    "Forex": ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"]
}

if asset_class == "Custom":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
else:
    ticker = st.sidebar.selectbox("Ticker Symbol", POPULAR_TICKERS[asset_class])
mode = st.sidebar.selectbox("Mode", ["Historical Analysis", "Live Forecasting"])

if mode == "Historical Analysis":
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    # Session State for Data
    if 'hist_data' not in st.session_state:
        st.session_state.hist_data = None
    if 'analysis_active' not in st.session_state:
        st.session_state.analysis_active = False
        
    if st.sidebar.button("Analyze"):
        with st.spinner("Fetching Data..."):
            df = fetch_data(ticker, period, interval)
            st.session_state.hist_data = df
            st.session_state.analysis_active = True
            # Reset LSTM state on new analysis
            st.session_state.lstm_trained = False
            st.session_state.lstm_preds = None
            st.session_state.lstm_actuals = None
            
    if st.session_state.analysis_active and st.session_state.hist_data is not None:
        df = st.session_state.hist_data
        
        if not df.empty:
            st.subheader(f"Historical Data for {ticker}")
            
            # Candlestick Chart
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='#0aff60',
                decreasing_line_color='#ff0a54',
                name=ticker)])
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="SF Pro Display, sans-serif", color="#a0a0b0"),
                height=600,
                xaxis=dict(showgrid=False, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators
            df = add_technical_indicators(df)
            st.write("Data with Indicators:", df.tail())
            
            # ARIMA Forecast
            st.subheader("ARIMA Forecast ♾️")
            with st.spinner("Running ARIMA..."):
                forecast = fit_predict_arima(df['Close'], steps=5)
                # Format forecast for display
                forecast_df = pd.DataFrame({"Forecast": forecast.values}, index=pd.date_range(start=df.index[-1], periods=5+1, freq='D')[1:])
                st.dataframe(forecast_df.style.format("${:.2f}"))
                
            # LSTM Forecast (Advanced)
            st.subheader("Deep Learning Forecast (LSTM)")
            
            if len(df) > 100:
                # Prepare multivariate data
                feature_cols = ['Close', 'RSI', 'MACD', 'MACD_Signal']
                data_for_model = df[feature_cols].dropna()
                
                # Session State for LSTM
                if 'lstm_trained' not in st.session_state:
                    st.session_state.lstm_trained = False
                    st.session_state.lstm_preds = None
                    st.session_state.lstm_actuals = None
                
                if st.button("Train Advanced LSTM Model") or st.session_state.lstm_trained:
                    if not st.session_state.lstm_trained:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner("Training Deep Learning Model... This may take a moment."):
                            # Normalize
                            scaler, scaled_data = normalize_data(data_for_model)
                            
                            # Create sequences
                            lookback = 60
                            X, y = prepare_lstm_data(scaled_data, lookback=lookback, target_col_index=0)
                            
                            # Split
                            train_size = int(len(X) * 0.8)
                            X_train, X_test = X[:train_size], X[train_size:]
                            y_train, y_test = y[:train_size], y[train_size:]
                            
                            # No reshape needed, X is already (samples, lookback, features)
                            
                            # Build Model
                            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                            
                            # Custom training loop
                            epochs = 20
                            batch_size = 32
                            
                            for epoch in range(epochs):
                                model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(f"Training Epoch {epoch + 1}/{epochs}")
                            
                            # Predict
                            preds = predict_lstm(model, X_test)
                            
                            # Inverse transform
                            dummy_array = np.zeros((len(preds), len(feature_cols)))
                            dummy_array[:, 0] = preds.flatten()
                            inverse_preds = scaler.inverse_transform(dummy_array)[:, 0]
                            
                            dummy_array_y = np.zeros((len(y_test), len(feature_cols)))
                            dummy_array_y[:, 0] = y_test.flatten()
                            inverse_actuals = scaler.inverse_transform(dummy_array_y)[:, 0]
                            
                            # Save to session state
                            st.session_state.lstm_trained = True
                            st.session_state.lstm_preds = inverse_preds
                            st.session_state.lstm_actuals = inverse_actuals
                            st.session_state.lstm_model = model # Optional: save model if needed
                            
                            status_text.text("Training Complete!")
                    
                    # Display Results (from session state)
                    if st.session_state.lstm_preds is not None:
                        chart_data = pd.DataFrame({
                            'Actual': st.session_state.lstm_actuals,
                            'Predicted': st.session_state.lstm_preds
                        })
                        
                        st.line_chart(chart_data, color=["#0aff60", "#00f2ff"])
                        
                        # Calculate metrics
                        mae = np.mean(np.abs(st.session_state.lstm_preds - st.session_state.lstm_actuals))
                        st.metric("Model MAE", f"${mae:.2f}")
                        
            else:
                st.warning("Not enough data to train LSTM. Please select a longer period (e.g., 1y or 2y).")

elif mode == "Live Forecasting":
    st.warning("Live Mode Active - Refreshing every 60 seconds")
    placeholder = st.empty()
    
    if "live_data" not in st.session_state:
        st.session_state.live_data = pd.DataFrame()

    # Simulation of live loop
    # In a real app, we might use st_autorefresh or just a manual loop with sleep if running locally
    
    if st.button("Start Live Feed"):
        try:
            while True:
                with placeholder.container():
                    price = get_latest_price(ticker)
                    st.metric(label=f"{ticker} Price", value=f"${price:.2f}")
                    
                    # Fetch recent 1m data for trend
                    df_live = fetch_data(ticker, period='1d', interval='1m')
                    if not df_live.empty:
                        # Custom Neon/Glass Chart Style
                        fig_live = go.Figure(data=[go.Candlestick(x=df_live.index,
                            open=df_live['Open'],
                            high=df_live['High'],
                            low=df_live['Low'],
                            close=df_live['Close'],
                            increasing_line_color='#0aff60', # Neon Green
                            decreasing_line_color='#ff0a54', # Neon Red
                            name=ticker
                        )])
                        
                        fig_live.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="SF Pro Display, sans-serif", color="#a0a0b0"),
                            height=500,
                            margin=dict(l=0, r=0, t=20, b=0),
                            xaxis=dict(
                                showgrid=False, 
                                showline=False, 
                                zeroline=False,
                                gridcolor='rgba(255,255,255,0.05)'
                            ),
                            yaxis=dict(
                                showgrid=True, 
                                gridcolor='rgba(255,255,255,0.05)',
                                zeroline=False,
                                showline=False
                            ),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_live, use_container_width=True)
                        
                        # Quick ARIMA on live data
                        close_series = df_live['Close']
                        if isinstance(close_series, pd.DataFrame):
                            close_series = close_series.iloc[:, 0]
                            
                        forecast_live = fit_predict_arima(close_series, steps=1)
                        
                        # Extract scalar value safely
                        pred_value = forecast_live.iloc[-1]
                        if isinstance(pred_value, pd.Series):
                            pred_value = pred_value.iloc[0]
                            
                        # Styled Prediction Banner
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, rgba(0, 242, 255, 0.1) 0%, rgba(0, 0, 0, 0) 100%);
                            border-left: 4px solid #00f2ff;
                            padding: 15px;
                            border-radius: 0 12px 12px 0;
                            margin-top: 20px;
                            backdrop-filter: blur(10px);
                        ">
                            <span style="color: #a0a0b0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">AI Forecast (Next Interval)</span><br>
                            <span style="color: #ffffff; font-size: 1.5rem; font-weight: 700;">${float(pred_value):.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)

                time.sleep(60)
                st.rerun()
        except KeyboardInterrupt:
            st.write("Stopped.")
