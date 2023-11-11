

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# Generate some sample data (replace with your own dataset)
sequence_length = 4
var_num = 3
input_data = np.random.rand(100, sequence_length, var_num)  # Three input variables
target_data = np.sin(np.sum(input_data, axis=1))  # Example target values
# 划分训练集和和测试集合
xTrain, xTest, yTrain, yTest = train_test_split(input_data, target_data, test_size=0.3, random_state=2)
# Create a Sequential model
model = keras.Sequential()

# Add the first LSTM layer
model.add(LSTM(units=32, input_shape=(sequence_length, var_num), activation='relu', return_sequences=True))

# Add the second LSTM layer
model.add(LSTM(units=32, activation='relu'))

# Add a Dense output layer
model.add(Dense(units=1))

# Define call back
# callbacks_set = [EarlyStopping(monitor='loss', baseline=0.01, verbose=1)]
callbacks_set = [EarlyStopping(monitor='loss', min_delta=0.0001, patience=60, mode='min', verbose=1)]

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics='mse')

# Train the model
model.fit(xTrain, yTrain, epochs=1000, batch_size=8, callbacks=callbacks_set)

# 模型评价
print(model.evaluate(xTest, yTest))
loss = model.evaluate(xTest, yTest)


# Evaluate the model if needed
# loss = model.evaluate(test_input_data, test_target_data)

# Make predictions
# predictions = model.predict(input_data)

# Print predictions
# print(predictions)



# # Train the model
# model.fit(input_data, input_data, validation_split=0.3, epochs=1000, batch_size=8, verbose=0, callbacks=callbacks_set)

# Make predictions
# predictions = model.predict(yData)
# mse = mean_squared_error(yData, predictions)