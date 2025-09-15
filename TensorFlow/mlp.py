import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

# In this tutorial, we will be training a lot of models. In order to use GPU memory cautiously,
# we will set tensorflow option to grow GPU memory allocation when required.
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# the following code creates a simple neural network model using Keras Sequential API.
# The model consists of three layers:
# 1. Flatten layer: This layer flattens the input images from a 2D array (28x28 pixels) to a 1D array (784 pixels).
# 2. Dense layer: This is a fully connected layer with 10 neurons and a softmax activation function.
#    The softmax activation function is used for multi-class classification problems, as it outputs a probability distribution over the classes.
# 3. The model is then summarized using the summary() method, which prints a summary of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.
# relu and sigmoid functions
# relu is used in hidden layers
# sigmoid is used in output layer for binary classification
# softmax is used in output layer for multi-class classification
def plot_function(f,name=''):
    plt.plot(range(-10,10), [f(tf.constant(x,dtype=tf.float32)) for x in range(-10,10)])
    plt.title(name)

plt.subplot(121)
plot_function(tf.nn.relu,'ReLU')
plt.subplot(122)
plot_function(tf.nn.sigmoid,'Sigmoid')
# plt.show()

# the above plot shows the ReLU and Sigmoid activation functions

model = keras.models.Sequential() # Create a Sequential model
model.add(keras.layers.Flatten(input_shape=(28,28))) # Flatten the input images (28x28) to 1D vectors (784)
model.add(keras.layers.Dense(100))     # 784 inputs, 100 outputs # Fully Connected Layer
model.add(keras.layers.ReLU())         # Activation Function # ReLU
model.add(keras.layers.Dense(10))      # 100 inputs, 10 outputs # Fully Connected Layer

model.summary()

# sparse_categorical_crossentropy is used when the labels are integers
# categorical_crossentropy is used when the labels are one-hot encoded vectors
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc']) 