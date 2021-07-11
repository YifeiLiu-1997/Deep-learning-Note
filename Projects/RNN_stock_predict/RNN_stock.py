"""
    RNN predict stock
    change SimpleRNN() -> LSTM() -> GRU()
"""
import os
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_path = 'stock/SH002362.csv'
print('----------load data----------')

if os.path.exists(data_path):
    print('load data success')
else:
    print('data dont exists, download now ...')
    df1 = ts.get_k_data('002362', ktype='D', start='2021-01-01', end='2021-07-10')
    df1.to_csv(data_path)

hanvon = pd.read_csv('stock/SH002362.csv')

# datasets
training_set = hanvon.iloc[0:100, 2:3].values
test_set = hanvon.iloc[100:, 2:3].values

# preprocessing
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(5, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 5:i, 0])
    y_train.append(training_set_scaled[i, 0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 5, 1))

for i in range(5, len(test_set)):
    x_test.append(test_set[i - 5:i, 0])
    y_test.append(test_set[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(20, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GRU(20),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='mean_squared_error')

check_point_path = 'checkpoint/RNN_stock.ckpt'
# if os.path.exists(check_point_path + '.index'):
#     print('----------load model----------')
#     model.load_weights(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=check_point_path,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss'
)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback],
                    epochs=20)

model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.title('Stock of maotai predict')
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.legend()
plt.show()

predicted_stock_price = model.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

real_stock_price = sc.inverse_transform(test_set[5:])

plt.plot(real_stock_price, color='red', label='Hanvon Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Hanvon Stock Price')
plt.title('Hanvon Stock Price Prediction')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
