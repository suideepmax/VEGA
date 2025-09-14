#Import the packages needed.
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

print(tf.__version__)

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data() #load the mnist dataset
#The dataset is split into training and testing sets. Each set contains images of handwritten digits (0-9).

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) # (10000, 28, 28) (10000,)

fig,ax = plt.subplots(1,7) #Create a figure with 1 row and 7 columns to display images
for i in range(7):
    ax[i].imshow(x_train[i]) #Display the i-th image from the training set
    ax[i].set_title(y_train[i]) #Set the title of the subplot to the corresponding label
    ax[i].axis('off') #Turn off the axis
plt.show() #Show the figure with the images

print('Training samples:',len(x_train)) #Print the number of training samples
print('Test samples:',len(x_test)) #Print the number of test samples

print('Tensor size:',x_train[0].shape) #Print the shape of the first training image
print('First 10 digits are:', y_train[:10]) #Print the first 10 labels from the training set
print('Type of data is ',type(x_train)) #Print the type of the training data

print('Min intensity value: ',x_train.min()) #Print the minimum intensity value in the training images
print('Max intensity value: ',x_train.max()) #Print the maximum intensity value in the training images

x_train = x_train.astype(np.float32)/255.0 #Normalize the training images to the range [0, 1]
x_test = x_test.astype(np.float32)/255.0 #Normalize the test images to the range [0, 1]

# the following code creates a simple neural network model using Keras Sequential API.
# The model consists of three layers:
# 1. Flatten layer: This layer flattens the input images from a 2D array (28x28 pixels) to a 1D array (784 pixels).
# 2. Dense layer: This is a fully connected layer with 10 neurons and a softmax activation function.
#    The softmax activation function is used for multi-class classification problems, as it outputs a probability distribution over the classes.
# 3. The model is then summarized using the summary() method, which prints a summary of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), 
        keras.layers.Dense(10,activation='softmax')])
model.summary()

print(model.layers[1].weights) #Get the weights of the second layer (Dense layer)

model.compile(optimizer='sgd',loss='categorical_crossentropy') #Compile the model with the SGD optimizer and categorical crossentropy loss function
# Note: Since the labels are integers, we should use sparse_categorical_crossentropy instead of categorical_crossentropy
#categorical_crossentropy is used when the labels are one-hot encoded vectors
# sparse_categorical_crossentropy is used when the labels are integers

y_train_onehot = keras.utils.to_categorical(y_train) #Convert the training labels to one-hot encoded vectors
y_test_onehot = keras.utils.to_categorical(y_test) #Convert the test labels to one-hot encoded vectors
print("First 3 training labels:",y_train[:3]) #Print the first 3 training labels
print("One-hot-encoded version:\n",y_train_onehot[:3]) #Print the one-hot-encoded version of the first 3 training labels
# one-hot encoding is a process of converting categorical variables into a binary matrix representation
# For example, if we have 3 classes (0, 1, 2), the one-hot encoding of the labels would be:
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

model.fit(x_train,y_train_onehot) #Train the model on the training data

# model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['acc']) #Compile the model with the SGD optimizer, categorical crossentropy loss function, and accuracy metric
# hist = model.fit(x_train,y_train_onehot,validation_data=(x_test,y_test_onehot), epochs=3) #Train the model on the training data with validation on the test data for 3 epochs
# An epoch is one complete pass through the entire training dataset

# the following code uses SGD optimizer with momentum to train the model for 5 epochs with a batch size of 64
# momentum is a technique that helps accelerate SGD in the relevant direction and dampens oscillations
# it does this by adding a fraction of the previous update to the current update
# this helps the optimizer to navigate along the relevant direction and avoid oscillations
# oscillations mean that the optimizer keeps changing direction and does not converge to the minimum
# the batch size is the number of samples that will be propagated through the network at once
model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), 
        keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer=keras.optimizers.SGD(momentum=0.5),loss='categorical_crossentropy',metrics=['acc'])
hist = model.fit(x_train,y_train_onehot,validation_data=(x_test,y_test_onehot), epochs=5, batch_size=64)  

for x in ['acc','val_acc']: #Plot training and validation accuracy over epochs
    plt.plot(hist.history[x])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train','val'])
plt.show()

# we do the following to visualize the weights of the first Dense layer
weight_tensor = model.layers[1].weights[0].numpy().reshape(28,28,10) # Get the weights of the Dense layer and reshape them for visualization
# The weight tensor has shape (28, 28, 10) because there are 10 neurons in the Dense layer, each connected to a 28x28 input image
fig, ax = plt.subplots(1,10,figsize=(15,4)) # Create a figure with 1 row and 10 columns to display the weights

# the following loop iterates over each of the 10 neurons in the Dense layer
# For each neuron, it extracts the corresponding weight matrix (of shape 28x28) and displays it as an image in a subplot
# the dark and light areas in the weight images represent the strength and polarity of the connections between the input pixels and the neuron
# dark areas represent negative weights, while light areas represent positive weights
# By visualizing the weights, we can gain insights into what features the model has learned to recognize for each class
# negative weights mean that the neuron is inhibited by the corresponding input pixel, while positive weights mean that the neuron is excited by the corresponding input pixel
for i in range(10):
    ax[i].imshow(weight_tensor[:,:,i]) # Display the weights for the i-th neuron
    ax[i].axis('off') # Turn off the axis
    ax[i].set_title(i)  # Set the title of the subplot to the corresponding class label
plt.show()