import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from tfcv import *
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# In this tutorial, we will be training a lot of models. In order to use GPU memory cautiously,
# we will set tensorflow option to grow GPU memory allocation when required.
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# the following code creates a simple convolutional neural network model using Keras Sequential API.
# The model consists of several layers:
# 1. Reshape layer: This layer reshapes the input images from a 2D array (28x28 pixels) to a 3D array (28x28x1) to include the channel dimension.
# 2. Conv2D layer: This is a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function.
#    The convolutional layer is used to extract features from the input images.
plot_convolution(x_train[:5],[[-1.,0.,1.],[-1.,0.,1.],[-1.,0.,1.]],'Vertical edge filter')
plot_convolution(x_train[:5],[[-1.,-1.,-1.],[0.,0.,0.],[1.,1.,1.]],'Horizontal edge filter')

# convolutional neural networks (CNNs) are particularly effective for image data because they can capture spatial hierarchies in images through the use of convolutional layers.
# layers in CNNs can learn to detect edges, textures, and more complex patterns in images, making them well-suited for tasks such as image classification, object detection, and image generation.
# The Conv2D layer applies a set of learnable filters to the input image, producing feature maps that highlight important features in the image.
# The Flatten layer then flattens the 3D feature maps into 1D vectors, which are then passed to a Dense layer for classification.
# The Dense layer is a fully connected layer with 10 neurons and no activation function (the logits are used directly for classification).
# The model is then summarized using the summary() method, which prints a summary of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=9, kernel_size=(5,5), input_shape=(28,28,1),activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])

model.summary()


# reshape the input data to include the channel dimension
# for grayscale images, the channel dimension is 1
# for color images, the channel dimension is 3
# the following code expands the dimensions of the input data to include the channel dimension
# the new shape of the input data will be (num_samples, 28, 28, 1)
x_train_c = np.expand_dims(x_train,3)
x_test_c = np.expand_dims(x_test,3)
hist = model.fit(x_train_c,y_train,validation_data=(x_test_c,y_test),epochs=7) # train the model for 3 epochs
plot_results(hist)
# plt.show()


fig,ax = plt.subplots(1,9)
l = model.layers[0].weights[0]
for i in range(9):
    ax[i].imshow(l[...,0,i])
    ax[i].axis('off')
plt.show()

# the following code makes use of two convolutional layers with max pooling in between to reduce the spatial dimensions of the feature maps.
# MaxPooling2D layers are used to downsample the feature maps, reducing their spatial dimensions and helping to control overfitting.
# The first Conv2D layer has 10 filters, a kernel size of 5x5, and a ReLU activation function.
# The second Conv2D layer has 20 filters, a kernel size of 5x5, and a ReLU activation function.
# After the convolutional and pooling layers, a Flatten layer is used to convert the 2D feature maps into a 1D vector.
# Finally, a Dense layer with 10 neurons is used as the output layer for classification.
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=10, kernel_size=(5,5), input_shape=(28,28,1), activation='relu'),
    keras.layers.MaxPooling2D(),  # Reduce feature map size
    keras.layers.Conv2D(filters=20, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPooling2D(),  # Further reduction
    keras.layers.Flatten(),       # Convert 2D feature maps into 1D vector
    keras.layers.Dense(10)        # Output layer (10 classes for digits 0–9)
])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])

model.summary()
hist = model.fit(x_train_c,y_train,validation_data=(x_test_c,y_test),epochs=3) # train the model for 3 epochs
plot_results(hist)
plt.show()