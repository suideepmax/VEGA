import os # For file path operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1' # Suppress TensorFlow logging
import tensorflow as tf
import numpy as np
import gzip

import random  
import matplotlib.pyplot as plt

labels_map = {
  0: 'T-Shirt',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle Boot',
}

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


# def_images basically does the same thing as tf.keras.datasets.fashion_mnist.load_data()
# def read_images(path: str, image_size: int, num_items: int) -> np.ndarray: # Read images from the given path
#     with gzip.open(path, 'rb') as f: # Open the gzip file
#         # a gzip file is a compressed file, so we need to decompress it first
#         data = np.frombuffer(f.read(), np.uint8, offset=16) # Read the data from the file, skipping the first 16 bytes (header)
#         data = data.reshape(num_items, image_size, image_size) # Reshape the data to (num_items, image_size, image_size)
#     return data

# def read_labels(path: str, num_items: int) -> np.ndarray: # Read labels from the given path
#   with gzip.open(path, 'rb') as file: # Open the gzip file
#     data = np.frombuffer(file.read(num_items + 8), np.uint8, offset=8) # Read the data from the file, skipping the first 8 bytes (header)
#     data = data.astype(np.int64) # Convert the data to int64
#   return data

# Parameters
# image_size = 28
# num_train = 60000
# num_test = 10000

# # Read the Fashion MNIST dataset
# training_images = read_images('data/FashionMNIST/raw/train-images-idx3-ubyte.gz', image_size, num_train)
# test_images = read_images('data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz', image_size, num_test)
# training_labels = read_labels('data/FashionMNIST/raw/train-labels-idx1-ubyte.gz', num_train)
# test_labels = read_labels('data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz', num_test)


# the following snippet displays 9 random images from the training set with their labels
figure = plt.figure(figsize=(8, 8)) # Create a figure to display images
cols = 3
rows = 3
# training_images is a numpy array of shape (60000, 28, 28)
for i in range(1, cols * rows + 1): # Display 9 random images from the training set
  sample_idx = random.randint(0, len(training_images) - 1) # Get a random index
  image = training_images[sample_idx] # Get the image at that index
  label = training_labels[sample_idx] # Get the label at that index
  figure.add_subplot(rows, cols, i) # Add a subplot to the figure
  plt.title(labels_map[label]) # Set the title to the label
  plt.axis('off') # Turn off the axis
  plt.imshow(image.squeeze(), cmap='gray') # Display the image in grayscale
plt.show() # Show the figure


# getting datasets 
# if you had a large dataset, you would need to wrap it in a tf.data.Dataset instance, which handles 
# large data better by making it easy to keep just a portion of it in memory. 
# You can wrap your data in a Dataset in this sample, so you're prepared to work with large data in the future.
# tf.data.Dataset.from_tensor_slices() creates a Dataset whose elements are slices of the given tensors.
# a tensor slice is a single element of the tensor along its first dimension.
# For example, if you have a tensor of shape (4, 28, 28),the slices are tensors of shape (28, 28).
# what is a tensor? A tensor is a generalization of vectors and matrices to potentially higher dimensions.
# A tensor can be thought of as a multi-dimensional array.
train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels)) # Create a TensorFlow dataset from the training images and labels
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)) # Create a TensorFlow dataset from the test images and labels

# in machine learning you want pixel values to be floating point values between 0 and 1
# so you need to normalize the images by dividing by 255.0
train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label)) # Normalize the images to [0, 1] range
test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label)) # Normalize the images to [0, 1] range

# Get a sample image as a NumPy array
# Now that you have a Dataset, you can no longer index it the same way as a NumPy array. Instead, you get an iterator by 
# calling the as_numpy_iterator method, and advance it by calling its next method. 
# At this point, you have a tuple containing an image and the corresponding label, so you can get the element at index 0 to inspect the image.
train_dataset.as_numpy_iterator().next()[0] # Get the first image from the training dataset as a NumPy array

# the following snippet shuffles the dataset and creates batches of 64 images
batch_size = 64 
train_dataset = train_dataset.batch(batch_size).shuffle(500)
test_dataset = test_dataset.batch(batch_size).shuffle(500)