# Tradesense-GPT
![Screenshot (1829)](https://github.com/Durgesh13kesharwani/Tradesense-GPT/assets/147710997/ac10eca5-8899-4aa6-b787-92abb85d15c6)
![Screenshot (1825)](https://github.com/Durgesh13kesharwani/Tradesense-GPT/assets/147710997/0f3f0a24-80b3-46cc-9f4c-2d1f560f5102)
![Screenshot (1826)](https://github.com/Durgesh13kesharwani/Tradesense-GPT/assets/147710997/312656a9-6d31-4edd-bedd-fd9832cdf95f)
![Screenshot (1827)](https://github.com/Durgesh13kesharwani/Tradesense-GPT/assets/147710997/e8822823-5bd7-43a8-82d0-807e37034539)

# 📈 TradeSense-GPT – Intelligent Stock Investing Made Simple

TradeSense-GPT is a powerful, intelligent, and interactive web application that empowers users to make smarter stock investment decisions. Built using Streamlit, Prophet, scikit-learn, and yfinance, this tool provides accurate stock forecasts, investment strategy simulations, and market trend analysis—all in a sleek, intuitive interface.

---

## 🚀 Features

- **Real-time Stock Data:** Fetches historical and current market data for popular stocks using Yahoo Finance.
- **Stock Price Forecasting:** Uses Facebook Prophet to forecast stock prices for 1 to 4 years.
- **Visual Analysis:** Interactive plots of historical and predicted prices using Plotly.
- **Investment Simulation:** Allows users to input investment amount and time period to estimate returns.
- **Market Intelligence Engine:** Incorporates sentiment analysis, fundamentals, and technical indicators to calculate:
  - Market trend (uptrend/downtrend)
  - Probability of profit/loss
  - Expected return amount
- **Model Accuracy Insights:** Trains a Random Forest Regressor to estimate prediction accuracy using historical features.
- **User Customization:** Includes payment method selector and adjustable parameters for simulation.

---

## 🛠️ Tech Stack

| Component | Description |
|----------|-------------|
| **Frontend** | Streamlit |
| **Data Source** | Yahoo Finance via `yfinance` |
| **Forecasting Model** | Prophet |
| **Machine Learning** | scikit-learn (RandomForestRegressor) |
| **Visualization** | Plotly, Matplotlib |
| **Language** | Python 3.x |

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Durgesh13kesharwani/TradeSense-GPT.git
cd TradeSense-GPT

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

1. Select a Stock:
   - Choose from BTC-USD, GOOG, AAPL, MSFT, or GME.

2. Adjust Forecast Horizon:
   - Use slider to choose 1 to 4 years.

3. View Data & Forecasts:
   - Explore raw and forecasted data with interactive plots.

4. Investment Simulation:
   - Enter investment amount and time period.

5. Analyze Returns:
   - View probabilities of profit/loss and expected return.

6. Model Accuracy:
   - See regression-based accuracy on historical data.

- Forecast Plots:
  Line chart of predicted vs actual closing prices

- Sidebar Inputs:
  Amount invested, time period, investment type

- Sidebar Outputs:
  Probability of profit, expected return %, and amount

- Payment Mode:
  UPI, Credit Card, Debit Card, PayPal, Bank Transfer

The app uses:

✅ Prophet for time-series prediction

✅ RandomForestRegressor for model accuracy estimation

✅ Market Trend Determination based on:
   • Bullish/Bearish market signals
   • Historical return & volatility
   • Company earnings growth & debt levels
   • RSI & moving averages from technical analysis
   • Sentiment from news and social media signals

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
