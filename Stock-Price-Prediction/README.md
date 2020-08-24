# Stock Price Prediction

## 1. Introduction

In this project, we will using machine learning to predict the returns on stocks. To build the stock price prediction model, we will take the stock data of Tata Global Beverages Limited as an example. Users can get the datasets from https://github.com/NickHan-cs/Deep_Learning_Datasets, and there are also the stock datasets of Apple, Facebook, Microsoft and Tesla. Then, we are going to predict stock price using the LSTM neural network.

在这个项目中，我们会利用机器学习去预测股票的回归。我们会用塔塔全球饮料公司的股票数据作为例子去构建股票预测模型。用户可以从https://github.com/NickHan-cs/Deep_Learning_Datasets上得到数据，同时那里还有苹果，脸书，微软和特斯拉的股票数据。接下来，我们要使用LSTM神经网络去预测股票价格了。

## 2. How to develop

### 2.1 Imports

* `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`: ignore the warning of importing tensorflow 忽略导入tensorflow库的警告信息
* ` rcParams['figure.figsize'] = 20, 10`: set default size of plots 设置图像的默认尺寸

```python
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
```

### 2.2 Read the datasets and analyse the closing prices

* `pd.to_datetime()`: convert the format of data to appointed date format of data 将数据格式转换成指定日期格式
*  `df.index=df['Date']` : After setting the index of dataframe as `df['Date']`, the x axis of plot will be `df['Date']` as default. 在将df的索引设置为`df['Date']`后，图像的x轴会默认为 `df['Date']`。

```python
df = pd.read_csv("D:/TensorFlow_datasets/NSE-Tata-Global-Beverages-Limited/NSE-Tata-Global-Beverages-Limited.csv")
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
print(df.head)
df.index = df['Date']
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')
plt.show()
```

### 2.3 Sort the dataset on date time and filter "Date" and "Close" columns

* `sort_index()`: ascending=True ascending order升序; ascending=False descending order 降序

```python
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]
```

### 2.4 Normalize the new filtered dataset

* `drop()`: inplace=True--modify the origin object 对原始对象进行修改; inplace=False--create a new object and modify 创建新的对象进行修改
* `new_data.values`: return array data of numpy type 返回numpy类型的array数据
* `MinMaxScaler(feature_range=(0, 1)).fit_transform(): Normalize the dataset 归一化数据集

In this project, to predict the stock price of one day, we use the stock prices of 60 days before the day. Therefore, the stock prices of 60 days before this day are the input of neural network, and the stock price of this day is the output.

在这个项目中，我们用前60天的股票价格来预测这天的股票价格。所以，这天前60天的股票价格是神经网络的输入，这天的股票价格是输出。

```python
new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)
final_dataset = new_data.values
train_dataset = final_dataset[0:987, :]
test_dataset = final_dataset[987:, :]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)
train_x, train_y = [], []
for i in range(60, len(train_dataset)):
    train_x.append(scaled_data[i - 60:i, 0])
    train_y.append(scaled_data[i, 0])
train_x, train_y = np.array(train_x), np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
train_x, train_y = shuffle(train_x, train_y)
```

### 2.5 Build and train the LSTM model

```python
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


RNN = LSTM()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = RNN.fit(train_x, train_y, epochs=20, batch_size=1, verbose=2, validation_split=0.1, callbacks=[reduce_lr, early_stopping])
```

### 2.6 Save the LSTM model

```python
RNN.save("LSTM_Stock_Price_Prediction_model")
```

### 2.7 Take a sample of a dataset to make stock price predictions using the LSTM model

* `scaler.transform()`: Normalize the test data 归一化测试集数据
* `scaler.transform()`: Inverse normalize the predicted price 逆归一化预测价格

```python
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
```

### 2.8 Visualize the predicted stock costs with the actual stock costs

```python
train_dataset = new_data[:987]
test_dataset = new_data[987:]
test_dataset['Predictions'] = predicted_close_price
plt.plot(train_dataset['Close'])
plt.plot(test_dataset[['Close', 'Predictions']])
plt.show()
```

## 3. Results

The running information are below.

运行信息如下所示。

```
Epoch 1/20
834/834 - 6s - loss: 9.8588e-04 - mean_squared_error: 9.8588e-04 - val_loss: 8.3744e-04 - val_mean_squared_error: 8.3744e-04 - lr: 0.0010
Epoch 2/20
834/834 - 6s - loss: 4.9368e-04 - mean_squared_error: 4.9368e-04 - val_loss: 7.2355e-04 - val_mean_squared_error: 7.2355e-04 - lr: 0.0010
Epoch 3/20
834/834 - 6s - loss: 4.0432e-04 - mean_squared_error: 4.0432e-04 - val_loss: 2.0646e-04 - val_mean_squared_error: 2.0646e-04 - lr: 0.0010
Epoch 4/20
834/834 - 6s - loss: 3.2083e-04 - mean_squared_error: 3.2083e-04 - val_loss: 1.2779e-04 - val_mean_squared_error: 1.2779e-04 - lr: 0.0010
Epoch 5/20
834/834 - 7s - loss: 2.8655e-04 - mean_squared_error: 2.8655e-04 - val_loss: 2.9990e-04 - val_mean_squared_error: 2.9990e-04 - lr: 0.0010
Epoch 6/20
834/834 - 6s - loss: 2.5640e-04 - mean_squared_error: 2.5640e-04 - val_loss: 1.1426e-04 - val_mean_squared_error: 1.1426e-04 - lr: 0.0010
Epoch 7/20
834/834 - 7s - loss: 1.8058e-04 - mean_squared_error: 1.8058e-04 - val_loss: 1.0532e-04 - val_mean_squared_error: 1.0532e-04 - lr: 2.0000e-04
Epoch 8/20
834/834 - 6s - loss: 1.8222e-04 - mean_squared_error: 1.8222e-04 - val_loss: 1.2388e-04 - val_mean_squared_error: 1.2388e-04 - lr: 2.0000e-04
Epoch 9/20
834/834 - 7s - loss: 1.8729e-04 - mean_squared_error: 1.8729e-04 - val_loss: 1.1678e-04 - val_mean_squared_error: 1.1678e-04 - lr: 2.0000e-04
Epoch 10/20
834/834 - 6s - loss: 1.8335e-04 - mean_squared_error: 1.8335e-04 - val_loss: 1.0487e-04 - val_mean_squared_error: 1.0487e-04 - lr: 2.0000e-04
Epoch 11/20
834/834 - 6s - loss: 1.6851e-04 - mean_squared_error: 1.6851e-04 - val_loss: 1.0494e-04 - val_mean_squared_error: 1.0494e-04 - lr: 4.0000e-05
Epoch 12/20
834/834 - 6s - loss: 1.6530e-04 - mean_squared_error: 1.6530e-04 - val_loss: 1.0330e-04 - val_mean_squared_error: 1.0330e-04 - lr: 4.0000e-05
Epoch 13/20
834/834 - 7s - loss: 1.6318e-04 - mean_squared_error: 1.6318e-04 - val_loss: 1.0661e-04 - val_mean_squared_error: 1.0661e-04 - lr: 4.0000e-05
Epoch 14/20
834/834 - 6s - loss: 1.6230e-04 - mean_squared_error: 1.6230e-04 - val_loss: 1.0427e-04 - val_mean_squared_error: 1.0427e-04 - lr: 8.0000e-06
Epoch 15/20
834/834 - 6s - loss: 1.6164e-04 - mean_squared_error: 1.6164e-04 - val_loss: 1.0376e-04 - val_mean_squared_error: 1.0376e-04 - lr: 8.0000e-06
Epoch 16/20
834/834 - 6s - loss: 1.6120e-04 - mean_squared_error: 1.6120e-04 - val_loss: 1.0633e-04 - val_mean_squared_error: 1.0633e-04 - lr: 8.0000e-06
Epoch 17/20
834/834 - 6s - loss: 1.6139e-04 - mean_squared_error: 1.6139e-04 - val_loss: 1.0399e-04 - val_mean_squared_error: 1.0399e-04 - lr: 1.6000e-06
```

The picture of comparisons between real stock prices and predicted stock prices and its zoom are belows. The curve of real stock prices is the yellow line, and the curve of predicted stock prices is the green line.

真实的股票价格和预测的股票价格的对比的图片及放大图如下所示。黄线是真实的股票价格曲线，绿线是预测的股票价格曲线。

![Stock_Price_Prediction.png](https://i.loli.net/2020/08/24/ODrG1utnbWlRaq4.png)

![Stock_Price_Prediction_Zoom.png](https://i.loli.net/2020/08/24/sEY6QHpR3UPo8VA.png)