# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:51:37 2018
@author: Sreenivas.J
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os

os.chdir("D:/Data Science/Data")
np.random.seed(100)

digit_train = pd.read_csv("Digits Recognizer_Train.csv")
digit_train.shape
digit_train.info()

#iloc[:, 1:] Means first to last row and 2nd column to last column
#255.0 --> Convert my data to 255 pixels
X_train = digit_train.iloc[:, 1:]/255.0
y_train = np_utils.to_categorical(digit_train["label"])
y_train.shape

#Here comes the basic Neural network
model = Sequential()
model.add(Dense(10, input_shape=(784,), activation='softmax'))
print(model.summary())

#mean_squared_error for regression
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, verbose=1, epochs=2, batch_size=2, validation_split=0.2)
print(model.get_weights())
