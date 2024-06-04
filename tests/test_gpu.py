import tensorflow as tf
import pandas as pd
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data = pd.read_csv('datasets/load.csv')

# Data preparation
# ==============================================================================
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data.set_index('Date')
data = data.asfreq('1D')
data = data.sort_index()
data.head(5)

data.head(5)

# Verify that the time series is complete
# ==============================================================================
(data.index == pd.date_range(start=data.index.min(),
                             end=data.index.max(),
                             freq=data.index.freq)).all()

print(f"Number of rows with missing values: {data.isnull().any(axis=1).mean()}")

# if not complete, fill with NaN values
if data.isnull().any(axis=1).mean() > 0.0:
    data.asfreq(freq='1D', fill_value=np.nan)

# Split the remaining data into train-validation-test
# ==============================================================================
data = data.loc['2006-01-01': '2019-12-31'].copy()
start_train = '2006-01-01'
end_train = '2018-12-31'
start_test = '2019-01-01'
data_train = data.loc[start_train:end_train, :].copy()
data_test  = data.loc[start_test:, :].copy()

print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Load'].values.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Prepare input sequences for the RNN model
def prepare_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


n_steps = 1850  # steps in the past used to predict steps in the future
sequence_length = n_steps
X_train, y_train = prepare_sequences(train_data, n_steps)
X_test, y_test = prepare_sequences(test_data, n_steps)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth( gpu, True )
      print("Num GPUs Available: ", len(gpus))
  except RuntimeError as e:
    print(e)

# Define the RNN model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(64, input_shape=(sequence_length, 1)))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Print the model summary
print(model.summary())


# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model on the testing set
loss = model.evaluate(X_test, y_test)
print('Test Loss:', loss)


from sklearn.metrics import mean_absolute_error, r2_score, max_error

# Make predictions on the testing set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# MEAN ABSOLUTE ERROR
mae = mean_absolute_error(actual_values, predictions)
print(f'Backtest error on test data (MAE): {mae}')

# R-squared = 1 - MSE/Var(y) (MSE: Mean Squared Error; Var(y): variance of dependent variable y)
r_squared = r2_score(actual_values, predictions)
print(f'Backtest error on test data (R-squared): {r_squared}')