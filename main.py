import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


weather_data = pd.read_csv('austin_weather.csv')
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

plt.plot(weather_data['Date'], weather_data['TempAvgF'])
plt.show()

weather_data_new = weather_data[['Date', 'TempAvgF']]
weather_data_new = weather_data_new.set_index('Date')


weather_value = weather_data_new.values
training_data_len = round(len(weather_value) * .7)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(weather_value)

prediction_days = 30


X_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], prediction_days, 1))
test_data = scaled_data[training_data_len - prediction_days:, :]


X_test = []
y_test = weather_value[training_data_len:, :]

for y in range(prediction_days, len(test_data)):
    X_test.append(test_data[y-prediction_days:y, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], prediction_days, 1))
"""
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

model.save("Forecast.h5")
"""
model = keras.models.load_model("Forecast.h5")

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

train = weather_data_new[:training_data_len]
Actual = weather_data_new[training_data_len:]
Actual['Predictions'] = predictions
plt.figure(figsize=(20, 9))
plt.title('Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Austin Weather', fontsize=18)
plt.plot(train['TempAvgF'])
plt.plot(Actual[['TempAvgF', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()
print(Actual)
