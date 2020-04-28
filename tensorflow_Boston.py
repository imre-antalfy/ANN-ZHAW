# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:23:58 2020

@author: imrea
"""

#%% import

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#%% function definitions

def norm(x):
    """used to norm the training data"""
    return (x - train_stats['mean']) / train_stats['std']

def plot_history(history):
    """ show the training process"""
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MEDV]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MEDV^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

#%% load data

(train_data, train_medv), (test_data, test_medv) = tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2, seed=113
)

train_data = pd.DataFrame(train_data)
train_medv = pd.DataFrame(train_medv)
test_data = pd.DataFrame(test_data)
test_medv = pd.DataFrame(test_medv)

#%% define classes
# see https://www.kaggle.com/c/boston-housing

class_names = ['crime','zoned_land','indus','river','nox_conc','av_rooms','age',
               'dis_center','acc_highways','tax','ptratio','black','lstat']

train_data.columns = [class_names]
test_data.columns = [class_names]
train_medv.columns = ['medv']
test_medv.columns = ['medv']

#%% Explore the data

sns.pairplot(train_data[["crime","age","dis_center","tax"]], diag_kind="kde")

# statistics of the data
train_stats = train_data.describe()
train_stats = train_stats.transpose()
train_stats


#%% preprocess data

#norm the data
normed_train_data = norm(train_data)
normed_test_data = norm(test_data)
normed_train_medv = train_medv / max(train_medv['medv'])
normed_test_medv = test_medv / max(test_medv['medv'])

#%% build model

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  
  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model();
model.summary()


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

#%% train the model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_medv,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

#%% visualize the learning process
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_history(history)

# # after 100 epochs, the training gets worse!!!!
# # stopping criteria needed

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_medv, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

#%% model prediction

test_predictions = model.predict(normed_test_data).flatten()
train_predictions = model.predict(normed_train_data).flatten()

plt.figure(3)
plt.scatter(test_medv, test_predictions)
plt.xlabel('True Values [MEDV]')
plt.ylabel('Predictions [MEDV]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

#%% error analysis

mse = mean_squared_error(test_medv, test_predictions)
print('Mean Squared Error: ',mse)
mae = mean_absolute_error(test_medv, test_predictions)
print('Mean Absolute Error: ',mae)
rsq = r2_score(train_medv,train_predictions) #R-Squared on the training data
print('R-square, Training: ',rsq)
rsq = r2_score(test_medv,test_predictions) #R-Squared on the testing data
print('R-square, Testing: ',rsq)
