# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:58:44 2018
@author: Sujatha Chittiri
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model #You need have GraphViz already installed in your machine
import os

def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))

os.chdir("D:/Data Science/Data")
np.random.seed(100)

digit_train = pd.read_csv("Digits Recognizer_Train.csv")
digit_train.shape
digit_train.info()

X_train = digit_train.iloc[:,1:]/255.0
y_train = np_utils.to_categorical(digit_train["label"])

model = Sequential()
#Hidden Layer1
model.add(Dense(512, input_shape=(784,), activation='sigmoid' ))
model.add(Dense(10,  input_shape=(512,), activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Plot Model
os.getcwd()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# =============================================================================
# #Visualizatio of network. But takes long time!!!
# from ann_visualizer.visualize import ann_viz;
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
# ann_viz(model, title = "model_NN.pdf", )
# =============================================================================

epochs = 20
batchsize = 5
history = model.fit(x=X_train, y=y_train, verbose=1, epochs=epochs, batch_size=batchsize, validation_split=0.2)
print(model.get_weights())
plot_loss_accuracy(history)

#Predictions on Test data
digit_test = pd.read_csv("Digits Recognizer_test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test.values.astype('float32')/255.0

pred = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred})
submissions.to_csv("submission_DigitsRec3.csv", index=False, header=True)
