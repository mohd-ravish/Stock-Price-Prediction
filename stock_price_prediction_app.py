import streamlit as st
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Title of the app
st.title("Stock Price Predictor App")

# Add a sidebar for user input
st.sidebar.header("User Input")
stock = st.sidebar.text_input("Enter the Stock ID", "MSFT")

# Display additional information
st.sidebar.markdown(
    """
    ## Additional Information
    - Go to [Yahoo Finance](https://finance.yahoo.com) to check more details about stocks.
    - Use this app to predict future stock prices based on historical data.
    """
)

# Fetch stock data
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
stock_data = yf.download(stock, start, end)

# Load the pre-trained model
model = load_model("stock_price_prediction_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(stock_data)

# Define the function to plot graphs


def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange', label='MA Values')
    plt.plot(full_data.Close, 'b', label='Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'g', label='Extra Data')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    return fig


# Plot Moving Averages
st.subheader('Original Close Price and MA for 250 days')
stock_data['MA_for_250_days'] = stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), stock_data['MA_for_250_days'], stock_data))

st.subheader('Original Close Price and MA for 200 days')
stock_data['MA_for_200_days'] = stock_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15, 6), stock_data['MA_for_200_days'], stock_data))

st.subheader('Original Close Price and MA for 100 days')
stock_data['MA_for_100_days'] = stock_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), stock_data['MA_for_100_days'], stock_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph(
    (15, 6), stock_data['MA_for_100_days'], stock_data, 1, stock_data['MA_for_250_days']))

# Prepare data for prediction
splitting_len = int(len(stock_data)*0.7)
x_test = pd.DataFrame(stock_data.Close[splitting_len:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare plotting data
plotting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=stock_data.index[splitting_len+100:])

# Display original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(plotting_data)

# Plot original vs predicted close price
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(
    pd.concat([stock_data.Close[:splitting_len+100], plotting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
