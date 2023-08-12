import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Fetch Bitcoin price data from CoinGecko API
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max&interval=daily"
response = requests.get(url)
data = response.json()

dates = [
    datetime.utcfromtimestamp(x[0] / 1000).strftime("%Y-%m-%d") for x in data["prices"]
]
prices = np.array(data["prices"])[:, 1].reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
prices_normalized = scaler.fit_transform(prices)

# Split data into training and test sets
train_size = int(len(prices_normalized) * 0.8)
train, test = prices_normalized[:train_size], prices_normalized[train_size:]

# Create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(train)):
    X_train.append(train[i - 60 : i, 0])
    y_train.append(train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(
            units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)
        ),
        tf.keras.layers.LSTM(units=50),
        tf.keras.layers.Dense(units=1),
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Prepare test data
X_test = []
y_test = test[60:]
for i in range(60, len(test)):
    X_test.append(test[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict prices
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Future predictions for next 30 days
future_days = 30
future_predictions = []
for i in range(future_days):
    x = test[-60:].reshape(1, 60, 1)
    future_price = model.predict(x)
    test = np.append(test, future_price).reshape(-1, 1)
    future_predictions.append(scaler.inverse_transform(future_price)[0][0])
    next_date = datetime.utcfromtimestamp(
        data["prices"][-1][0] / 1000 + i * 86400
    ).strftime("%Y-%m-%d")
    dates.append(next_date)

# Visualize the results
plt.figure(figsize=(15, 6))

# Flatten the prices array to make it 1D
flattened_prices = prices[train_size + 60 :].flatten()

# Combine real prices and predicted prices for plotting
combined_dates = dates[train_size + 60 :]
combined_prices = np.concatenate((flattened_prices, future_predictions))
combined_predicted_prices = np.concatenate(
    (predicted_prices.flatten(), future_predictions)
)
plt.plot(combined_dates, combined_prices, color="red", label="Real Bitcoin Price")
plt.plot(
    combined_dates,
    combined_predicted_prices,
    color="blue",
    label="Predicted Bitcoin Price",
)
plt.title("Bitcoin Price Prediction")
plt.xlabel("Date")
plt.ylabel("Bitcoin Price")
plt.xticks(
    combined_dates[::60], rotation=45
)  # Show every 60th date for clarity and rotate labels for better visibility
plt.legend()
plt.tight_layout()
plt.show()
