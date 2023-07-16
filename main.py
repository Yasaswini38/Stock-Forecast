import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction')

stocks = ['AAPL', 'GOOGL', 'MSFT', 'GME', 'AMZN', 'NFLX', 'META', 'DIS']
selected_stock = st.selectbox('Select dataset for prediction', stocks)

start_dates = {
    'AAPL': "1980-12-12",
    'GOOGL': "2004-08-19",
    'MSFT': "1986-03-13",
    'GME': "2002-02-13",
    'AMZN': "1997-05-15",
    'NFLX': "2002-05-23",
    'META': "2021-10-28",
    'DIS': "1962-01-02"
}

START = start_dates[selected_stock]
TODAY = date.today().strftime("%Y-%m-%d")

n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to train the model and calculate performance metrics
def train_model(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_features = train_data[['Open', 'High', 'Low', 'Volume']]
    train_labels = train_data['Close']
    test_features = test_data[['Open', 'High', 'Low', 'Volume']]
    test_labels = test_data['Close']
    model = RandomForestRegressor()
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_labels, predictions)
    mae = mean_absolute_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    
    # Calculate the loss for the current stock
    loss = mse + mae
    
    return test_labels, predictions, mse, mae, r2, loss

# Load the data
data = load_data(selected_stock)

# Call the train_model function and get the necessary values
test_labels, predictions, mse, mae, r2, loss = train_model(data)

# Display the regression evaluation metrics
st.sidebar.subheader('Regression Evaluation Metrics')
st.sidebar.write('Mean Squared Error (MSE):', mse)
st.sidebar.write('Mean Absolute Error (MAE):', mae)
st.sidebar.write('R-squared (R2) Score:', r2)

# Display the raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Create and fit the Prophet model with a changepoint_prior_scale of 0.05
m = Prophet(changepoint_prior_scale=0.15)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig1.update_layout(title='Stock Prediction')
st.plotly_chart(fig1)

# Plot forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Calculate performance metrics
actual_values = data['Close'].values[-period:]
predicted_values = forecast['yhat'].values[-period:]

mape = (abs(actual_values - predicted_values) / actual_values).mean() * 100
rmse = ((actual_values - predicted_values) ** 2).mean() ** 0.5
mae = abs(actual_values - predicted_values).mean()

# Display performance metrics in the sidebar
st.sidebar.subheader('Performance Metrics')
st.sidebar.write('MAPE:', round(mape, 2))
st.sidebar.write('RMSE:', round(rmse, 2))
st.sidebar.write('MAE:', round(mae, 2))

# Sidebar section
st.sidebar.title('Performance Metrics')
st.sidebar.subheader('Select metrics thresholds')

# Generate unique keys for each slider widget
mape_threshold = st.sidebar.slider('Mean Absolute Percentage Error (MAPE)', 0.0, 100.0, 10.0, key='mape_threshold')
rmse_threshold = st.sidebar.slider('Root Mean Square Error (RMSE)', 0.0, 100.0, 10.0, key='rmse_threshold')
mae_threshold = st.sidebar.slider('Mean Absolute Error (MAE)', 0.0, 100.0, 10.0, key='mae_threshold')

# Calculate profit/loss matrix for all stocks
best_stock = None
min_loss = float('inf')
for stock in stocks:
    stock_data = load_data(stock)
    _, _, _, _, _, loss = train_model(stock_data)
    if loss < min_loss:
        min_loss = loss
        best_stock = stock

# Display the best stock
st.sidebar.subheader('Best Stock to Invest')
st.sidebar.write(best_stock)
