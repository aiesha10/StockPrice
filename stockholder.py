# Stock Price Prediction using LSTM
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------
# Step 1: Download AAPL stock data
# -------------------------------
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
data.to_csv("AAPL.csv")
print("Data saved as AAPL.csv")
print(data.head())

# -------------------------------
# Step 2: Preprocessing
# -------------------------------
# Use only relevant features: Open, High, Low, Close, Volume
df = data[['Open','High','Low','Close','Volume']].values

scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Train-test split
training_size = int(len(df_scaled) * 0.65)
train_data = df_scaled[:training_size]
test_data = df_scaled[training_size:]

# Create dataset for LSTM
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step])
        Y.append(dataset[i+time_step, 3])  # Predict Close price
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# -------------------------------
# Step 3: Build Stacked LSTM Model
# -------------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# -------------------------------
# Step 4: Predictions
# -------------------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to original scale
train_predict_inv = scaler.inverse_transform(
    np.concatenate([np.zeros((train_predict.shape[0],3)), train_predict, np.zeros((train_predict.shape[0],1))], axis=1)
)[:,3]

test_predict_inv = scaler.inverse_transform(
    np.concatenate([np.zeros((test_predict.shape[0],3)), test_predict, np.zeros((test_predict.shape[0],1))], axis=1)
)[:,3]

# -------------------------------
# Step 5: Plotting Train/Test predictions
# -------------------------------
close_price = df[:,3]  # Original Close price
train_plot = np.full_like(close_price, np.nan)
train_plot[time_step:len(train_predict_inv)+time_step] = train_predict_inv

test_plot = np.full_like(close_price, np.nan)
test_start = len(train_predict_inv) + time_step
test_plot[test_start:test_start+len(test_predict_inv)] = test_predict_inv

plt.figure(figsize=(12,6))
plt.plot(close_price, label='Original Close Price')
plt.plot(train_plot, label='Train Predict')
plt.plot(test_plot, label='Test Predict')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price USD')
plt.legend()
plt.show()

# -------------------------------
# Step 6: Predict Future 30 Days
# -------------------------------
future_steps = 30
temp_input = test_data[-time_step:].tolist()  # Last 60 days from test_data
future_output = []

for i in range(future_steps):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 5)
    yhat = model.predict(x_input)
    temp = list(x_input[0])
    temp[time_step-1][3] = yhat  # Replace last Close value
    temp_input.append(temp[-1])
    future_output.append(yhat[0,0])

# Inverse transform future predictions
future_output_inv = scaler.inverse_transform(
    np.concatenate([np.zeros((future_steps,3)), np.array(future_output).reshape(-1,1), np.zeros((future_steps,1))], axis=1)
)[:,3]

plt.figure(figsize=(12,6))
plt.plot(range(len(close_price)), close_price, label='Original Close Price')
plt.plot(range(len(close_price), len(close_price)+future_steps), future_output_inv, label='Future 30 Days Prediction')
plt.title('AAPL Future 30 Days Prediction')
plt.xlabel('Time')
plt.ylabel('Price USD')
plt.legend()
plt.show()
