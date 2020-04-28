# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:24:41 2020

@author: imrea
"""


#%% import

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#%% load data

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#%% define classes

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%% Explore the data

train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

#%% preprocess data

# data contained in the images needs to be rescaled to range from 0 to 1
plt.figure(1)
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# rescale
train_images = train_images / 255.0
test_images = test_images / 255.0

# check rescale
plt.figure(2)
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# take a look at data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#%% build model

# 
model = keras.Sequential([
    # reformatting layer, from square to line representation
    keras.layers.Flatten(input_shape=(28, 28)),
    # fully connected layer, 128 neurons
    keras.layers.Dense(128, activation='relu'),
    # fully connected layer, 10 neurons
    keras.layers.Dense(10)
])

# compiling the model
    #loss function, optimizer, metrics
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%% training model

# feed data
# model learn to associate labls and images
# model should make predictions about a test set
# verify the prediction with the test set labels

# feed the data
model.fit(train_images, train_labels, epochs=10)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# compared to the accuracy of the training dataset, the test set accuracy is 
# lower





















