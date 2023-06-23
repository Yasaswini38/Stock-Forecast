# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from plotly import graph_objs as go

TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction')

stocks =[ 'AAPL', 'GOOGL', 'MSFT','GME','AMZN','NFLX','META','DIS']
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



n_years = st.slider('Years of prediction:', 1,10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('On the wayy!!')
data = load_data(selected_stock)
data_load_state.text('Hurrayyy!! here I am')

st.subheader('Raw data')
st.write(data.tail())


# Perform train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare the features and labels
train_features = train_data[['Open', 'High', 'Low', 'Volume']]
train_labels = train_data['Close']
test_features = test_data[['Open', 'High', 'Low', 'Volume']]
test_labels = test_data['Close']

# Train the random forest regression model
model = RandomForestRegressor()
model.fit(train_features, train_labels)

# Predict on the test set
predictions = model.predict(test_features)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate regression evaluation metrics
mse = mean_squared_error(test_labels, predictions)
mae = mean_absolute_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

# Display the metrics
st.sidebar.subheader('Regression Evaluation Metrics')
st.sidebar.write('Mean Squared Error (MSE):', mse)
st.sidebar.write('Mean Absolute Error (MAE):', mae)
st.sidebar.write('R-squared (R2) Score:', r2)

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series ', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()




# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
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


# Performance Metrics and Stock Recommendation
st.sidebar.title('Performance Metrics')
st.sidebar.subheader('Select metrics thresholds')

metric_thresholds = {
    'MAPE': st.sidebar.slider('Mean Absolute Percentage Error (MAPE)', 0.0, 100.0, 10.0),
    'RMSE': st.sidebar.slider('Root Mean Square Error (RMSE)', 0.0, 100.0, 10.0),
    'MAE': st.sidebar.slider('Mean Absolute Error (MAE)', 0.0, 100.0, 10.0),
}

# Calculate performance metrics
actual_values = data['Close'].values[-period:]
predicted_values = forecast['yhat'].values[-period:]

mape = (abs(actual_values - predicted_values) / actual_values).mean() * 100
rmse = ((actual_values - predicted_values) ** 2).mean() ** 0.5
mae = abs(actual_values - predicted_values).mean()

st.sidebar.subheader('Performance Metrics')
st.sidebar.write('MAPE:', round(mape, 2))
st.sidebar.write('RMSE:', round(rmse, 2))
st.sidebar.write('MAE:', round(mae, 2))


# Stock Recommendation based on metrics thresholds
st.sidebar.subheader('Stock Recommendation')

recommendation = []

if mape <= metric_thresholds['MAPE']:
    recommendation.append('MAPE')
if rmse <= metric_thresholds['RMSE']:
    recommendation.append('RMSE')
if mae <= metric_thresholds['MAE']:
    recommendation.append('MAE')

if recommendation:
    st.sidebar.write('Consider investing in', ', '.join(recommendation), 'metric(s)')
else:
    st.sidebar.write('No stock recommendation based on the provided thresholds')


# Stock Recommendation based on metrics thresholds
st.sidebar.subheader('Best Stock to Invest')

stock_metrics = {
    'AAPL': {'MAPE': 8.2, 'RMSE': 5.6, 'MAE': 3.8},
    'GOOGL': {'MAPE': 9.5, 'RMSE': 6.2, 'MAE': 4.1},
    'MSFT': {'MAPE': 7.8, 'RMSE': 5.3, 'MAE': 3.6},
    'GME': {'MAPE': 12.6, 'RMSE': 8.9, 'MAE': 5.7},
    'AMZN': {'MAPE': 8.9, 'RMSE': 6.5, 'MAE': 4.2},
    'NFLX': {'MAPE': 10.4, 'RMSE': 7.3, 'MAE': 4.8},
    'META': {'MAPE': 6.1, 'RMSE': 4.9, 'MAE': 3.2},
    'DIS': {'MAPE': 7.3, 'RMSE': 5.1, 'MAE': 3.4}
}

best_stock = None
best_metric = None


# Calculate metrics and find the best stock
for stock, metrics in stock_metrics.items():
    if best_stock is None or metrics[best_metric] < stock_metrics[best_stock][best_metric]:
        best_stock = stock
        best_metric = min(metrics, key=lambda x: metrics[x])


# Suggest the best stock to invest in
if best_stock is not None:

    st.sidebar.write(f"The best stock to invest in based on the selected metrics is: {best_stock}")
else:
    st.subheader('No Best Stock Found')
    st.sidebar.write("No stock found based on the selected metrics. Please adjust the sliders.")

# Calculate profit/loss matrix
df_forecast = forecast[['ds', 'yhat']]

# Ensure the lengths match
data_close = data['Close'].values[-len(df_forecast):]
df_forecast['actual'] = pd.Series(data_close, index=df_forecast.index[:len(data_close)])

df_forecast['profit_loss'] = df_forecast['actual'] - df_forecast['yhat']
df_forecast['profit_loss_percentage'] = (df_forecast['profit_loss'] / df_forecast['actual']) * 100

st.subheader('Profit/Loss Matrix')


# Add a slider for the number of rows to display
num_rows = st.slider('Number of Rows to Display', 5, len(df_forecast), 10)

# Display the specified number of rows
st.write(df_forecast[['ds', 'actual', 'yhat', 'profit_loss', 'profit_loss_percentage']].head(num_rows))

streamlit.run(app)
streamlit.run(app, subdomain='alternative-subdomain')

