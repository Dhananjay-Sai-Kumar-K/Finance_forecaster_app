import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from src.data_loader import fetch_data, get_latest_price, fetch_live_data
from src.preprocessing import add_technical_indicators, normalize_data, prepare_lstm_data
from src.models.arima import fit_predict_arima
from src.models.lstm import build_lstm_model, train_lstm_model, predict_lstm
from src.models.rl_agent import StockTradingEnv, train_rl_agent, evaluate_agent

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

mode = st.sidebar.selectbox("Mode", ["Historical Analysis", "Portfolio Comparison", "Live Forecasting", "Reinforcement Learning"])

# Theme Toggle
theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
if theme == "Light":
    st.markdown("""
    <style>
        :root {
            --glass-bg: rgba(240, 240, 250, 0.5);
            --text-primary: #1a1a2e;
            --text-secondary: #4a4a6a;
            --glass-border: rgba(0, 0, 0, 0.1);
        }
        
        /* Main Background */
        .stApp {
            background-color: #f8f9fe;
        }
        
        /* Sidebar container */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Typography overrides */
        .stMarkdown, p, h1, h2, h3, li, .stMetricLabel {
            color: #1a1a2e !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #4a4a6a !important;
        }
        div[data-testid="stMetricValue"] {
            color: #1a1a2e !important;
        }
        
        /* Button Styling for Light Mode */
        .stButton > button {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
            border: 1px solid #e0e0eb !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #f0f0ff !important;
            border-color: #6c5ce7 !important;
            color: #6c5ce7 !important;
            transform: translateY(-2px);
        }
        
        /* Inputs & Selectboxes */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #f8f9fa !important;
            color: #1a1a2e !important;
            border-color: #e0e0eb !important;
        }
        
        /* Dropdown Text */
        div[data-baseweb="select"] span {
            color: #1a1a2e !important;
        }
        
        /* Dropdown Menu Items */
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }
        
        /* Multiselect styling */
        span[data-baseweb="tag"] {
            background-color: #e0e0eb !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Advanced Settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    st.write("**LSTM Parameters**")
    lstm_epochs = st.slider("Epochs", 5, 50, 20, help="Number of training iterations")
    lstm_lookback = st.slider("Lookback Window", 30, 120, 60, help="Number of past time steps to use")
    
    st.write("**ARIMA Parameters**")
    arima_p = st.slider("AR Order (p)", 1, 10, 5, help="Autoregressive order")
    arima_d = st.slider("Differencing (d)", 0, 2, 1, help="Degree of differencing")
    arima_q = st.slider("MA Order (q)", 0, 5, 0, help="Moving average order")

if mode == "Historical Analysis":
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    # Session State for Data
    if 'hist_data' not in st.session_state:
        st.session_state.hist_data = None
    if 'analysis_active' not in st.session_state:
        st.session_state.analysis_active = False
        
    if st.sidebar.button("Analyze"):
        try:
            with st.spinner("Fetching Data..."):
                df = fetch_data(ticker, period, interval)
                
                # Validate data
                if df.empty:
                    st.error(f"❌ No data found for ticker **{ticker}**. Please verify the ticker symbol.")
                    st.info("💡 Tip: Use Yahoo Finance format (e.g., BTC-USD for Bitcoin, ^GSPC for S&P 500)")
                    st.stop()
                
                if len(df) < 10:
                    st.warning(f"⚠️ Only {len(df)} data points found. Try a longer period for better analysis.")
                
                st.session_state.hist_data = df
                st.session_state.analysis_active = True
                # Reset LSTM state on new analysis
                st.session_state.lstm_trained = False
                st.session_state.lstm_preds = None
                st.session_state.lstm_actuals = None
                st.session_state.lstm_metrics = None
                st.success(f"✅ Successfully loaded {len(df)} data points for {ticker}")
                
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.exception(e)
            st.stop()
            
    if st.session_state.analysis_active and st.session_state.hist_data is not None:
        df = st.session_state.hist_data
        
        if not df.empty:
            st.subheader(f"Historical Data for {ticker}")
            
            # Technical Indicators
            df = add_technical_indicators(df)
            
            # Enhanced Chart with Subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f"{ticker} Price", "RSI"),
                                row_heights=[0.7, 0.3])

            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='#0aff60',
                decreasing_line_color='#ff0a54',
                name=ticker), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#a0a0b0')), row=2, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="SF Pro Display, sans-serif", color="#a0a0b0"),
                height=700, # Increased height
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_xaxes(showgrid=False, gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators
            df = add_technical_indicators(df)
            st.write("Data with Indicators:", df.tail())
            
            # Download raw data with indicators
            csv_data = df.to_csv()
            st.download_button(
                label="📥 Download Data with Indicators (CSV)",
                data=csv_data,
                file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # ARIMA Forecast
            st.subheader("ARIMA Forecast ♾️")
            with st.spinner("Running ARIMA..."):
                # Use custom parameters from sliders
                arima_order = (arima_p, arima_d, arima_q)
                forecast, arima_metrics = fit_predict_arima(df['Close'], order=arima_order, steps=5)
                
                # Store metrics in session state for comparison
                st.session_state.arima_metrics = arima_metrics
                
                # Format forecast for display
                forecast_df = pd.DataFrame({"Forecast": forecast.values}, index=pd.date_range(start=df.index[-1], periods=5+1, freq='D')[1:])
                st.dataframe(forecast_df.style.format("${:.2f}"))
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"${arima_metrics['MAE']:.2f}")
                col2.metric("RMSE", f"${arima_metrics['RMSE']:.2f}")
                col3.metric("Order", f"({arima_metrics['Order'][0]},{arima_metrics['Order'][1]},{arima_metrics['Order'][2]})")
                
                # Download forecast
                forecast_csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download ARIMA Forecast (CSV)",
                    data=forecast_csv,
                    file_name=f"{ticker}_arima_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            # LSTM Forecast (Advanced)
            st.subheader("Deep Learning Forecast (LSTM)")
            
            if len(df) > 100:
                # Prepare multivariate data
                # Updated with new indicators
                feature_cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'EMA_20', 'SMA_50']
                # Ensure all columns exist
                available_cols = [c for c in feature_cols if c in df.columns]
                data_for_model = df[available_cols].dropna()
                
                # Session State for LSTM
                if 'lstm_trained' not in st.session_state:
                    st.session_state.lstm_trained = False
                    st.session_state.lstm_preds = None
                    st.session_state.lstm_actuals = None
                    st.session_state.lstm_metrics = None
                
                # Model Persistence Path
                model_dir = "models"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                # Create a unique filename based on ticker, period, and lookback
                model_filename = f"{model_dir}/lstm_{ticker}_{period}_lb{lstm_lookback}.keras"
                model_exists = os.path.exists(model_filename)
                
                train_btn = st.button("Train Advanced LSTM Model")
                load_btn = False
                if model_exists and not st.session_state.lstm_trained:
                    st.success(f"✅ Found pre-trained model for {ticker}")
                    load_btn = st.button("Load Saved Model")
                
                if train_btn or load_btn or st.session_state.lstm_trained:
                    if not st.session_state.lstm_trained:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner("Processing Deep Learning Model..."):
                            # Normalize
                            scaler, scaled_data = normalize_data(data_for_model)
                            
                            # Create sequences with custom lookback
                            X, y = prepare_lstm_data(scaled_data, lookback=lstm_lookback, target_col_index=0)
                            
                            # Split
                            train_size = int(len(X) * 0.8)
                            X_train, X_test = X[:train_size], X[train_size:]
                            y_train, y_test = y[:train_size], y[train_size:]
                            
                            model = None
                            
                            # Load existing model if requested
                            if load_btn and model_exists:
                                try:
                                    from tensorflow.keras.models import load_model
                                    model = load_model(model_filename)
                                    status_text.text("Model Loaded Successfully!")
                                    progress_bar.progress(100)
                                except Exception as e:
                                    st.error(f"Error loading model: {e}")
                                    model = None
                            
                            # Train new model if needed
                            if model is None:
                                # Build Model
                                model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                                
                                # Custom training loop with custom epochs
                                batch_size = 32
                                
                                for epoch in range(lstm_epochs):
                                    model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
                                    progress_bar.progress((epoch + 1) / lstm_epochs)
                                    status_text.text(f"Training Epoch {epoch + 1}/{lstm_epochs}")
                                
                                # Save the model
                                model.save(model_filename)
                                status_text.text("Training Complete & Model Saved!")
                            
                            # Predict
                            preds = predict_lstm(model, X_test)
                            
                            # Inverse transform
                            dummy_array = np.zeros((len(preds), len(available_cols)))
                            dummy_array[:, 0] = preds.flatten()
                            inverse_preds = scaler.inverse_transform(dummy_array)[:, 0]
                            
                            dummy_array_y = np.zeros((len(y_test), len(available_cols)))
                            dummy_array_y[:, 0] = y_test.flatten()
                            inverse_actuals = scaler.inverse_transform(dummy_array_y)[:, 0]
                            
                            # Calculate metrics
                            mae = np.mean(np.abs(inverse_preds - inverse_actuals))
                            rmse = np.sqrt(np.mean((inverse_preds - inverse_actuals) ** 2))
                            
                            # Save to session state
                            st.session_state.lstm_trained = True
                            st.session_state.lstm_preds = inverse_preds
                            st.session_state.lstm_actuals = inverse_actuals
                            st.session_state.lstm_metrics = {'MAE': mae, 'RMSE': rmse}
                            
                    # Display Results (from session state)
                    if st.session_state.lstm_preds is not None:
                        chart_data = pd.DataFrame({
                            'Actual': st.session_state.lstm_actuals,
                            'Predicted': st.session_state.lstm_preds
                        })
                        
                        st.line_chart(chart_data, color=["#0aff60", "#00f2ff"])
                        
                        # Display metrics
                        metrics = st.session_state.lstm_metrics
                        col1, col2 = st.columns(2)
                        col1.metric("LSTM MAE", f"${metrics['MAE']:.2f}")
                        col2.metric("LSTM RMSE", f"${metrics['RMSE']:.2f}")
                        
                        # Download predictions
                        pred_csv = chart_data.to_csv()
                        st.download_button(
                            label="📥 Download LSTM Predictions (CSV)",
                            data=pred_csv,
                            file_name=f"{ticker}_lstm_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
            else:
                st.warning("Not enough data to train LSTM. Please select a longer period (e.g., 1y or 2y).")

            # 📊 Model Comparison Dashboard
            # Check if both metrics exist AND are not None
            if st.session_state.get('arima_metrics') is not None and st.session_state.get('lstm_metrics') is not None:
                st.markdown("---")
                st.subheader("📊 Model Comparison")
                
                arima_m = st.session_state.arima_metrics
                lstm_m = st.session_state.lstm_metrics
                
                comparison_data = {
                    'Model': ['ARIMA', 'LSTM'],
                    'MAE': [f"${arima_m['MAE']:.2f}", f"${lstm_m['MAE']:.2f}"],
                    'RMSE': [f"${arima_m['RMSE']:.2f}", f"${lstm_m['RMSE']:.2f}"],
                    'MAE_Raw': [arima_m['MAE'], lstm_m['MAE']] # For highlighting
                }
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Determine winner
                winner = "ARIMA" if arima_m['MAE'] < lstm_m['MAE'] else "LSTM"
                st.success(f"🏆 Best Performing Model: **{winner}** (Lower MAE)")
                
                st.table(comp_df[['Model', 'MAE', 'RMSE']])

elif mode == "Portfolio Comparison":
    st.header("📊 Portfolio Comparison")
    
    # Flatten popular tickers list for options
    all_tickers = [t for sublist in POPULAR_TICKERS.values() for t in sublist]
    
    # Multi-select for tickers
    default_tickers = ["AAPL", "MSFT", "GOOGL"]
    selected_tickers = st.multiselect("Select Assets to Compare", 
                                      options=list(set(all_tickers)), # Unique values
                                      default=default_tickers)
    
    if st.button("Compare Performance"):
        if not selected_tickers:
            st.warning("Please select at least one ticker.")
        else:
            comparison_data = {}
            metrics_list = []
            
            with st.spinner("Fetching Portfolio Data..."):
                for t in selected_tickers:
                    # Fetch 1y data for comparison
                    df = fetch_data(t, period='1y', interval='1d')
                    if not df.empty:
                        # Normalize for comparison (start at 0%)
                        start_price = df['Close'].iloc[0]
                        norm_price = ((df['Close'] - start_price) / start_price) * 100
                        comparison_data[t] = norm_price
                        
                        # Metrics
                        current_price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                        change = ((current_price - prev_price) / prev_price) * 100
                        
                        metrics_list.append({
                            "Ticker": t,
                            "Price": f"${current_price:.2f}",
                            "24h Change": f"{change:+.2f}%",
                            "Volatility (30d)": f"{df['Close'].pct_change().rolling(30).std().iloc[-1]*100:.2f}%"
                        })
            
            if comparison_data:
                # Chart
                st.subheader("Relative Performance (1 Year Return %)")
                st.line_chart(pd.DataFrame(comparison_data))
                
                # Metrics Table
                st.subheader("Asset Metrics")
                st.dataframe(pd.DataFrame(metrics_list))
            else:
                st.error("No data available for selected tickers.")

elif mode == "Live Forecasting":
    st.info("⚡ Live Mode Active - Auto-refreshing every 60 seconds")
    
    # Auto-refresh
    count = st_autorefresh(interval=60000, limit=None, key="live_refresh")
    
    # Main content
    price = get_latest_price(ticker)
    st.metric(label=f"{ticker} Price", value=f"${price:.2f}")
    
    # Fetch recent data
    df_live = fetch_live_data(ticker)
    
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
            
        # Use custom parameters if set, otherwise default
        # Note: arima_p, arima_d, arima_q are available from sidebar
        forecast_live, _ = fit_predict_arima(close_series, order=(arima_p, arima_d, arima_q), steps=1)
        
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

elif mode == "Reinforcement Learning":
    st.header("🤖 Reinforcement Learning Trading Agent")
    st.write("Train a PPO (Proximal Policy Optimization) agent to trade stocks.")
    
    if st.button("Start Training Agent"):
        df = fetch_data(ticker, period='2y', interval='1d')
        
        if not df.empty:
            # Prepare environment
            env_data = add_technical_indicators(df)
            env_data = env_data.dropna()
            
            if len(env_data) > 100:
                env = StockTradingEnv(env_data)
                
                # Train
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Training RL agent... This may take a few minutes"):
                    # Train for 10000 timesteps for demo purposes
                    agent = train_rl_agent(env, timesteps=10000)
                    progress_bar.progress(100)
                    status_text.text("Training Complete!")
                    
                    # Evaluate
                    st.subheader("Agent Performance")
                    history = evaluate_agent(agent, env)
                    
                    # Visualize Net Worth
                    st.line_chart(history['net_worth'])
                    
                    # Calculate final return
                    initial_balance = 10000
                    final_balance = history['net_worth'][-1]
                    roi = ((final_balance - initial_balance) / initial_balance) * 100
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Initial Balance", f"${initial_balance:,.2f}")
                    col2.metric("Final Balance", f"${final_balance:,.2f}", f"{roi:.2f}%")
                    
            else:
                st.warning("Not enough data to train RL agent. Need at least 100 data points.")
        else:
            st.error("Failed to fetch data for training.")
