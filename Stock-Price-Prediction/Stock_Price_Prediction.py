import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


def LSTM():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=64, return_sequences=True),
        tf.keras.layers.LSTM(units=64),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


# Read the dataset
df = pd.read_csv("D:/TensorFlow_datasets/NSE-Tata-Global-Beverages-Limited/NSE-Tata-Global-Beverages-Limited.csv")
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
# print(df.head)
# Analyze the closing prices from dataframe
df.index = df['Date']
'''
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')
plt.show()
'''
# Sort the dataset on date time and filter "Date" and "Close" columns
data = df.sort_index(ascending=True, axis=0)  # ascending True-升序
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]
# Normalize the new filtered dataset
new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)  # inplace True-对原始对象进行修改 False-创建新的对象进行修改
final_dataset = new_data.values  # 返回numpy类型的ndarray数据
train_dataset = final_dataset[0:987, :]
test_dataset = final_dataset[987:, :]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)  # 归一化
train_x, train_y = [], []
for i in range(60, len(train_dataset)):
    train_x.append(scaled_data[i - 60:i, 0])
    train_y.append(scaled_data[i, 0])
train_x, train_y = np.array(train_x), np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
train_x, train_y = shuffle(train_x, train_y)
# Build and train the LSTM model
RNN = LSTM()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = RNN.fit(train_x, train_y, epochs=20, batch_size=1, verbose=2, validation_split=0.1,
                  callbacks=[reduce_lr, early_stopping])
# save the LSTM model
RNN.save("LSTM_Stock_Price_Prediction_model")
# Take a Sample of a dataset to make the stock price predictions using the LSTM model
test_data = new_data[len(new_data) - len(test_dataset) - 60:].values
test_data = test_data.reshape(-1, 1)
test_data = scaler.transform(test_data)
test_x = []
for i in range(60, test_data.shape[0]):
    test_x.append(test_data[i - 60:i, 0])
test_x = np.array(test_x)
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
predicted_close_price = RNN.predict(test_x)
predicted_close_price = scaler.inverse_transform(predicted_close_price)
# Visualize the predicted stock costs with the actual stock costs
train_dataset = new_data[:987]
test_dataset = new_data[987:]
test_dataset['Predictions'] = predicted_close_price
plt.plot(train_dataset['Close'])
plt.plot(test_dataset[['Close', 'Predictions']])
plt.show()
