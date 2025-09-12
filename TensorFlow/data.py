import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from typing import Tuple

from kintro import *

learning_rate = 0.1 # Learning rate for the optimizer
batch_size = 64 # Batch size for training

(train_dataset, test_dataset) = get_data(batch_size) # Get the training and testing datasets

model = NeuralNetwork() # Create an instance of the NeuralNetwork model

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # Loss function for training
optimizer = tf.keras.optimizers.SGD(learning_rate) # Stochastic Gradient Descent optimizer
metrics = ['accuracy'] # Metrics to evaluate during training and testing
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics) # Compile the model with the optimizer, loss function, and metrics

# we need epochs because the dataset is small and the model is simple
# epochs is the number of times the model will see the entire dataset
# batch size is the number of samples the model will see before updating the weights
# higher epochs means the model will see the entire dataset more times
# less loss if more epochs
epochs = 5 # Number of epochs to train the model
print('\nFitting:') 
model.fit(train_dataset, epochs=epochs) # Train the model on the training dataset

# Evaluate the model on the testing dataset
# the evaluate method returns the loss and accuracy of the model on the testing dataset
# we can also use the evaluate method to get the loss and accuracy of the model on the training dataset
print('\nEvaluating:')
(test_loss, test_accuracy) = model.evaluate(test_dataset)
print(f'\nTest accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}') # Evaluate the model on the testing dataset and print the accuracy and loss

os.makedirs("outputs", exist_ok=True) # Create the 'outputs' directory if it doesn't exist
model.save('outputs/model.keras') # Save the trained model to the 'outputs/model' directory