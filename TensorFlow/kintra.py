import os # For file path operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1' # Suppress TensorFlow logging
import tensorflow as tf
import numpy as np
import gzip
from typing import Tuple

class NeuralNetwork(tf.keras.Model): # Define a neural network model by subclassing tf.keras.Model
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.sequence = tf.keras.Sequential([ # Define the layers of the model using tf.keras.Sequential
      tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten the input images from 28x28 to a 784-dimensional vector
      tf.keras.layers.Dense(128, activation='relu'), # First dense layer with 128 units and ReLU activation
      tf.keras.layers.Dense(20, activation='relu'), # Second dense layer with 20 units and ReLU activation
      tf.keras.layers.Dense(10) # Output layer with 10 units (one for each class)
    ])

  def call(self, x: tf.Tensor) -> tf.Tensor: # Define the forward pass of the model
    # input x is a batch of images with shape (batch_size, 28, 28)
    y_prime = self.sequence(x) # Pass the input through the layers
    return y_prime # Return the output logits
  
def get_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  # Load the Fashion MNIST dataset from TensorFlow Keras datasets
  (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  # Create TensorFlow datasets from the training and test images and labels
  train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  # Normalize the images to [0, 1] range by dividing by 255.0
  train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
  test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

  # Define batch size
  batch_size = 64 

  # Shuffle and batch the datasets
  train_dataset = train_dataset.batch(batch_size).shuffle(500)
  test_dataset = test_dataset.batch(batch_size).shuffle(500)

  return train_dataset, test_dataset

