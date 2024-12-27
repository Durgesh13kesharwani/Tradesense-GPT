import streamlit as st
import yfinance as yf
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
# from patch_prophet_plot import patch_plot_plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Function to load stock data
def load_data(ticker):
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)
    return data

# Function to predict stock prices
def predict_stock(data, n_years):
    # Prepare the dataframe for Prophet
    df_train = data.reset_index()[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
    # Ensure 'y' contains numeric values
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert non-numeric to NaN
    df_train = df_train.dropna(subset=['y'])  # Drop rows with NaN values
    
    # Fit the Prophet model
    m = Prophet()
    m.fit(df_train)
    
    # Make future dataframe and predictions
    future = m.make_future_dataframe(periods=n_years * 365)
    forecast = m.predict(future)
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
st.title('[TradeSense GPT]- Intelligent Stock Investing Made Simple')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load stock data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
st.subheader('Raw data plot')
st.line_chart(data['Close'])

# Predict and plot forecast
st.subheader('Forecast data')
forecast = predict_stock(data, n_years)
st.write(forecast.tail())

st.subheader(f'Forecast plot for {n_years} years')
fig = plot_stock_forecast(forecast)
st.plotly_chart(fig)



# Function to determine market trend based on analysis parameters
def determine_market_trend(market_conditions, historical_performance, fundamentals, technical_analysis, sentiment_analysis):
    # Determine market trend based on analysis parameters
    if market_conditions['bullish'] and market_conditions['volatility'] == 'low' \
        and historical_performance['average_returns'] > 0 and historical_performance['std_dev'] < 0.1 \
        and fundamentals['earnings_growth'] == 'high' and fundamentals['debt_levels'] == 'low' \
        and technical_analysis['moving_average'] == 'above' and technical_analysis['RSI'] == 'overbought' \
        and sentiment_analysis['positive_news'] and sentiment_analysis['social_media_sentiment'] == 'bullish':
        return 'uptrend'
    else:
        return 'downtrend'

# Function to parse time period input and convert to months
def parse_time_period(time_input):
    time_input = time_input.lower()
    if 'year' in time_input:
        return int(time_input.split()[0]) * 12
    elif 'month' in time_input:
        return int(time_input.split()[0])
    elif 'day' in time_input:
        return int(time_input.split()[0]) / 30  # Assuming 30 days in a month
    else:
        return 0

# Take user input for investment amount and time period
st.sidebar.title('Investment Inputs')
amount_invested = st.sidebar.number_input("Amount Invested (Rs)", value=10000.0, step=100.0)
time_input = st.sidebar.text_input("Time Period (year/month/day)", value="1 year")
time_period = parse_time_period(time_input)

# Function to calculate probability of profit and loss based on parameters
def calculate_probabilities(market_conditions, historical_performance, fundamentals, technical_analysis, sentiment_analysis, model_accuracy, amount_invested, time_period, investment_type):
    # Adjust investment type based on time period
    if time_period >= 1:
        investment_type = 'long term'
    else:
        investment_type = 'short term'

    # Determine market trend based on analysis parameters
    market_trend = determine_market_trend(market_conditions, historical_performance, fundamentals, technical_analysis, sentiment_analysis)

    # Placeholder values for demonstration - Replace with actual calculations
    base_probability_of_profit = 0.5
    base_probability_of_loss = 0.5

    # Adjust probabilities based on additional parameters and market trend
    if market_trend == 'uptrend':
        if model_accuracy < 0.8:
            base_probability_of_profit -= 0.1
            base_probability_of_loss += 0.1
        
        if amount_invested > 10000:
            base_probability_of_profit += 0.05
            base_probability_of_loss -= 0.05
        
        if investment_type == 'long term':
            base_probability_of_profit += 0.05
            base_probability_of_loss -= 0.05
        elif investment_type == 'short term':
            base_probability_of_profit -= 0.05
            base_probability_of_loss += 0.05

    elif market_trend == 'downtrend':
        if model_accuracy < 0.8:
            base_probability_of_profit += 0.1
            base_probability_of_loss -= 0.1
        
        if amount_invested > 10000:
            base_probability_of_profit -= 0.05
            base_probability_of_loss += 0.05
        
        if investment_type == 'long term':
            base_probability_of_profit -= 0.05
            base_probability_of_loss += 0.05
        elif investment_type == 'short term':
            base_probability_of_profit += 0.05
            base_probability_of_loss -= 0.05

    # Placeholder calculations for simplicity
    probability_of_profit = base_probability_of_profit * 0.8  # Adjusted based on model confidence
    probability_of_loss = base_probability_of_loss * 0.8  # Adjusted based on model confidence

    return probability_of_profit, probability_of_loss, investment_type

# Function to calculate expected return percentage
def calculate_expected_return(probability_of_profit, probability_of_loss, gpt_model_charge, other_charges, taxes, market_trend):
    # Calculate expected return percentage based on market trend
    if market_trend == 'uptrend':
        expected_return_percentage = probability_of_profit - gpt_model_charge - other_charges - taxes
    elif market_trend == 'downtrend':
        expected_return_percentage = probability_of_loss - gpt_model_charge - other_charges - taxes
    else:
        expected_return_percentage = 0  # No clear trend, no expected return

    return max(expected_return_percentage, 0)  # Ensure positive return

# Example parameters
market_conditions = {'bullish': True, 'volatility': 'low'}
historical_performance = {'average_returns': 0.08, 'std_dev': 0.05}
fundamentals = {'earnings_growth': 'high', 'debt_levels': 'low'}
technical_analysis = {'moving_average': 'above', 'RSI': 'overbought'}
sentiment_analysis = {'positive_news': True, 'social_media_sentiment': 'bullish'}
accuracy = 99.99679049316039
model_accuracy = accuracy  # Example model accuracy


# Adjust investment type based on time period
if time_period >= 1:
    investment_type = 'long term'
else:
    investment_type = 'short term'

# Determine market trend based on analysis parameters
market_trend = determine_market_trend(market_conditions, historical_performance, fundamentals, technical_analysis, sentiment_analysis)

# Example charges and taxes
gpt_model_charge = 0.05
other_charges = 0.1
taxes = 0.05

# Calculate probabilities
probability_of_profit, probability_of_loss, investment_type = calculate_probabilities(market_conditions, historical_performance, fundamentals, technical_analysis, sentiment_analysis, model_accuracy, amount_invested, time_period, None)

# Calculate expected return percentage
expected_return_percentage = calculate_expected_return(probability_of_profit, probability_of_loss, gpt_model_charge, other_charges, taxes, market_trend)


# Print results
st.sidebar.write("Probability of Profit:", probability_of_profit)
st.sidebar.write("Probability of Loss:", probability_of_loss)
st.sidebar.write("Investment Type:", investment_type)

# Calculate the expected return amount
expected_return_amount = amount_invested * (1 + expected_return_percentage) ** (time_period / 12)

# Display expected return percentage and amount
st.sidebar.write(f"Expected Return Percentage: {expected_return_percentage * 100:.2f}%")
st.sidebar.write(f"Expected Return Amount: Rs {expected_return_amount:.2f}")

# Payment option
st.sidebar.subheader("Payment Options")
payment_options = ["UPI","Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
payment_method = st.sidebar.selectbox("Select Payment Method", payment_options)

st.sidebar.write("You have selected:", payment_method)

# Stock prediction accuracy
st.subheader('Stock Prediction Accuracy')

# Load the dataset
data = yf.download(selected_stock, start="2015-01-01", end=date.today().strftime("%Y-%m-%d"))

# Extract features and target
X = data.drop(columns=["Close"])
Y = data["Close"]

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)

# Predict on test data
test_data_prediction = regressor.predict(X_test)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)

# Choose time period
time_periods = {"1 year": "1y", "1 month": "1mo", "3 months": "3mo","1 day": "1d" , "5 years": "5y"}
selected_period = st.selectbox("Select Time Period", list(time_periods.keys()))

# Plot actual data as candlesticks
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name='Actual'))

# Plot predicted data
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(color='green'), name="Predicted"))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='green'), name="Upper Bound"))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(color='red'), name="Lower Bound"))

fig.layout.update(title_text='Stock Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)


# Calculate mean absolute percentage error (MAPE)
mape = metrics.mean_absolute_percentage_error(Y_test, test_data_prediction)

# Calculate accuracy
accuracy = 100 - mape

# Print accuracy
st.write("Accuracy:", accuracy, "%")
