import streamlit as st
import yfinance as yf
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Function to load stock data using yfinance
def load_data(ticker):
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    try:
        data = yf.download(ticker, start=START, end=TODAY, auto_adjust=False)
        if data.empty:
            raise ValueError("No data found.")
        data.reset_index(inplace=True)

        # Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(col).strip() if col[1] else col[0] for col in data.columns]
        else:
            data.columns = [col.strip() for col in data.columns]  # Clean any whitespace

        return data
    except Exception as e:
        st.error(f"Failed to retrieve data for {ticker}. Error: {e}")
        return pd.DataFrame()

# Function to predict stock prices
def predict_stock(data, n_years):
    # Try to find the correct 'Adj Close' column dynamically
    adj_close_col = next((col for col in data.columns if 'Adj Close' in col), None)
    if not adj_close_col or 'Date' not in data.columns:
        raise ValueError("Data must contain 'Date' and 'Adj Close' columns.")

    df_train = data[['Date', adj_close_col]].copy()
    df_train.columns = ['ds', 'y']
    df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    df_train.dropna(subset=['ds', 'y'], inplace=True)

    if df_train.empty or len(df_train) < 2:
        raise ValueError("Not enough valid rows for forecasting.")

    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=n_years * 365)
    forecast = model.predict(future)
    return forecast

# Function to plot stock forecast
def plot_stock_forecast(forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Close Price"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name="Lower Bound"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name="Upper Bound"))
    fig.layout.update(title_text='Stock Forecast', xaxis_rangeslider_visible=True)
    return fig

# Streamlit app
st.title('[TradeSense GPT] - Intelligent Stock Investing Made Simple')

stocks = ('BTC-USD','GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load stock data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

if data.empty:
    st.error("Stock data could not be loaded.")
    st.stop()

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
st.subheader('Raw data plot')
adj_close_col = next((col for col in data.columns if 'Adj Close' in col), None)
if adj_close_col:
    st.line_chart(data[adj_close_col])
else:
    st.warning("'Adj Close' column not found.")

# Forecast section
st.subheader('Forecast data')
forecast = predict_stock(data, n_years)
st.write(forecast.tail())
st.subheader(f'Forecast plot for {n_years} years')
fig = plot_stock_forecast(forecast)
st.plotly_chart(fig)

# Sidebar Inputs
st.sidebar.title('Investment Inputs')
amount_invested = st.sidebar.number_input("Amount Invested (Rs)", value=10000.0, step=100.0)
time_input = st.sidebar.text_input("Time Period (year/month/day)", value="1 year")

def parse_time_period(time_input):
    time_input = time_input.lower()
    if 'year' in time_input:
        return int(time_input.split()[0]) * 12
    elif 'month' in time_input:
        return int(time_input.split()[0])
    elif 'day' in time_input:
        return int(time_input.split()[0]) / 30
    else:
        return 0

time_period = parse_time_period(time_input)

# Market analysis & return estimation
def determine_market_trend(mc, hp, f, ta, sa):
    if mc['bullish'] and mc['volatility'] == 'low' \
        and hp['average_returns'] > 0 and hp['std_dev'] < 0.1 \
        and f['earnings_growth'] == 'high' and f['debt_levels'] == 'low' \
        and ta['moving_average'] == 'above' and ta['RSI'] == 'overbought' \
        and sa['positive_news'] and sa['social_media_sentiment'] == 'bullish':
        return 'uptrend'
    return 'downtrend'

def calculate_probabilities(mc, hp, f, ta, sa, acc, amt, tp, inv_type):
    inv_type = 'long term' if tp >= 1 else 'short term'
    trend = determine_market_trend(mc, hp, f, ta, sa)
    p_profit = 0.5

    if trend == 'uptrend':
        if acc < 0.8: p_profit -= 0.1
        if amt > 10000: p_profit += 0.05
        if inv_type == 'long term': p_profit += 0.05
        else: p_profit -= 0.05
    else:
        if acc < 0.8: p_profit += 0.1
        if amt > 10000: p_profit -= 0.05
        if inv_type == 'long term': p_profit -= 0.05
        else: p_profit += 0.05

    return p_profit * 0.8, (1 - p_profit) * 0.8, inv_type

def calculate_expected_return(p_profit, p_loss, gpt_charge, other_charges, taxes, trend):
    return max((p_profit if trend == 'uptrend' else p_loss) - gpt_charge - other_charges - taxes, 0)

# Simulated parameters
market_conditions = {'bullish': True, 'volatility': 'low'}
historical_performance = {'average_returns': 0.08, 'std_dev': 0.05}
fundamentals = {'earnings_growth': 'high', 'debt_levels': 'low'}
technical_analysis = {'moving_average': 'above', 'RSI': 'overbought'}
sentiment_analysis = {'positive_news': True, 'social_media_sentiment': 'bullish'}

accuracy = 99.99679049316039
model_accuracy = accuracy / 100
gpt_model_charge = 0.05
other_charges = 0.1
taxes = 0.05

prob_profit, prob_loss, inv_type = calculate_probabilities(
    market_conditions, historical_performance, fundamentals,
    technical_analysis, sentiment_analysis,
    model_accuracy, amount_invested, time_period, None
)

exp_return_percent = calculate_expected_return(prob_profit, prob_loss, gpt_model_charge, other_charges, taxes,
                                               determine_market_trend(market_conditions, historical_performance,
                                                                      fundamentals, technical_analysis,
                                                                      sentiment_analysis))
exp_return_amt = amount_invested * (1 + exp_return_percent) ** (time_period / 12)

# Sidebar Outputs
st.sidebar.write("Probability of Profit:", round(prob_profit, 2))
st.sidebar.write("Probability of Loss:", round(prob_loss, 2))
st.sidebar.write("Investment Type:", inv_type)
st.sidebar.write(f"Expected Return Percentage: {exp_return_percent * 100:.2f}%")
st.sidebar.write(f"Expected Return Amount: Rs {exp_return_amt:.2f}")

st.sidebar.subheader("Payment Options")
payment_method = st.sidebar.selectbox("Select Payment Method", ["UPI", "Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
st.sidebar.write("You have selected:", payment_method)

# Stock Prediction Accuracy
st.subheader('Stock Prediction Accuracy')

feature_cols = [col for col in data.columns if any(x in col for x in ['Open', 'High', 'Low', 'Volume'])]

if not adj_close_col or len(feature_cols) < 4 or 'Date' not in data.columns:
    st.warning("Missing required columns for prediction.")
else:
    X = data[feature_cols]
    Y = data[adj_close_col]

    if len(X) < 10:
        st.warning("Not enough data to train prediction model.")
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train, Y_train.values.ravel())

        test_prediction = regressor.predict(X_test)

        r2 = metrics.r2_score(Y_test, test_prediction)
        mape = metrics.mean_absolute_percentage_error(Y_test, test_prediction)
        accuracy = 100 - mape * 100

        # Plot actual data as candlesticks
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data['Date'],
                                     open=data[feature_cols[0]],
                                     high=data[feature_cols[1]],
                                     low=data[feature_cols[2]],
                                     close=data[adj_close_col], name='Actual'))

        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(color='green'), name="Predicted"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='green'), name="Upper Bound"))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(color='red'), name="Lower Bound"))

        fig.layout.update(title_text='Stock Forecast', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

        st.write(f"Prediction Accuracy: {accuracy:.2f}%")
