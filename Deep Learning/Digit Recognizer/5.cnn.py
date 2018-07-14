# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:12:06 2018
@author: Sujatha Chittiri
"""

from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
#import utils

os.chdir("D:/Data Science/Data")
np.random.seed(100)

digit_train = pd.read_csv("Digits Recognizer_Train.csv")
digit_train.shape
digit_train.info()

X_train = digit_train.iloc[:,1:].values.astype('float32')/255.0
X_train_images=X_train.reshape(X_train.shape[0],28,28,1)
y_train = np_utils.to_categorical(digit_train["label"])

img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
#Add 1st level of convulution network with 3X3 filter, Activation = ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#Apply Max pooling for pre-dominent feature recoginiton
model.add(MaxPooling2D((2, 2)))

#Add 2 level of convulution network with 3X3 filter, Activation = ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
#Apply Max pooling for pre-dominent feature recoginiton
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
#Now we extracted some featrues

#Apply regular FFNN on extracted features for classification
model.add(Dense(512, activation='relu'))
#Output classifications(In this case 0, 1,2 ....9)
model.add(Dense(10,  activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
batchsize = 16
history = model.fit(x=X_train_images, y=y_train, verbose=1, epochs=epochs, batch_size=batchsize, validation_split=0.2)
print(model.get_weights())

historydf = pd.DataFrame(history.history, index=history.epoch)
utils.plot_loss_accuracy(history)
