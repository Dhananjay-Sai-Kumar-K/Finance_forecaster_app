# Finance Forecaster App - Quick Start Guide

## ✅ Status: Ready to Run!

All dependencies are installed and working correctly.

## 🚀 How to Run

### Option 1: Using Command Line

1. Open PowerShell/Terminal
2. Navigate to project directory:
   ```bash
   cd "e:\Academics\DSK College Activities\projects\Finance_forecaster_app"
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Your browser will automatically open to `http://localhost:8501`

### Option 2: Using VS Code

1. Open the integrated terminal (Ctrl + `)
2. Run:
   ```bash
   streamlit run app.py
   ```

## 📖 Usage Instructions

### Historical Analysis Mode (Recommended for First Use)

1. **Select Asset Class**: Choose from Stocks, Crypto, Commodities, Indices, or Forex
2. **Pick a Ticker**: 
   - Stocks: Try AAPL (Apple), MSFT (Microsoft), or GOOGL (Google)
   - Crypto: Try BTC-USD (Bitcoin) or ETH-USD (Ethereum)
3. **Set Time Period**: 
   - For LSTM training: Use "1y" or "2y" (needs >100 data points)
   - For quick analysis: Use "1mo" or "3mo"
4. **Click "Analyze"**: Wait for data to load
5. **View Results**:
   - Interactive candlestick chart
   - Technical indicators (RSI, MACD)
   - ARIMA 5-day forecast
6. **Train LSTM** (optional):
   - Click "Train Advanced LSTM Model"
   - Wait ~30-60 seconds for training
   - View predictions vs actual prices
   - Check MAE (Mean Absolute Error) metric

### Live Forecasting Mode

1. **Switch Mode**: Select "Live Forecasting" from sidebar
2. **Choose Ticker**: Pick any asset
3. **Click "Start Live Feed"**: 
   - Shows real-time price
   - Updates every 60 seconds
   - Displays ARIMA forecast for next interval
4. **Stop**: Press Ctrl+C in terminal or close browser tab

## 🎯 Example Workflows

### Workflow 1: Analyze Apple Stock
```
1. Asset Class: Stocks
2. Ticker: AAPL
3. Period: 1y
4. Interval: 1d
5. Click "Analyze"
6. Click "Train Advanced LSTM Model"
7. Compare ARIMA vs LSTM predictions
```

### Workflow 2: Track Bitcoin Live
```
1. Mode: Live Forecasting
2. Asset Class: Crypto
3. Ticker: BTC-USD
4. Click "Start Live Feed"
5. Watch real-time updates
```

### Workflow 3: Compare Multiple Assets
```
1. Analyze AAPL (save screenshot)
2. Change ticker to MSFT
3. Click "Analyze" again
4. Compare results
```

## ⚠️ Important Notes

- **LSTM Training**: Requires at least 100 data points (use 1y or 2y period)
- **Live Mode**: Currently uses blocking loop - may freeze UI
- **First Run**: May take 10-15 seconds to load TensorFlow
- **Internet Required**: Fetches data from Yahoo Finance API

## 🐛 Troubleshooting

### Issue: "No data found for ticker"
**Solution**: Check ticker symbol spelling. Use Yahoo Finance format (e.g., BTC-USD for Bitcoin)

### Issue: "Not enough data to train LSTM"
**Solution**: Select a longer period (1y or 2y) instead of 1mo

### Issue: Live mode freezes
**Solution**: This is a known issue. Use Historical Analysis mode instead, or restart the app

### Issue: Slow performance
**Solution**: 
- Close other applications
- Reduce LSTM epochs (edit line 141 in app.py)
- Use shorter time periods

## 📊 Understanding the Output

### ARIMA Forecast
- Shows next 5 days of predicted prices
- Based on statistical patterns
- Fast but simple

### LSTM Predictions
- Shows predicted vs actual prices on test data
- Uses deep learning
- More accurate but slower
- MAE shows average error in dollars

### Technical Indicators
- **RSI**: 0-100 scale (>70 = overbought, <30 = oversold)
- **MACD**: Trend indicator (positive = bullish, negative = bearish)

## 🎨 UI Features

- **Neon Green Candles**: Price went up
- **Neon Red Candles**: Price went down
- **Glassmorphic Sidebar**: Frosted glass effect
- **Dark Theme**: Easy on the eyes

## 📝 Next Steps

After running the app, check out the full walkthrough document for:
- Detailed architecture explanation
- Improvement recommendations
- Code optimization tips
- Advanced features to add

## 🆘 Need Help?

1. Check the walkthrough.md for detailed documentation
2. Review error messages in terminal
3. Verify all dependencies are installed: `pip list`
4. Try with a different ticker or time period

---

**Ready to forecast? Run `streamlit run app.py` now!** 🚀
