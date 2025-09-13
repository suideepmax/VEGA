import gzip
import numpy as np
import tensorflow as tf
from typing import Tuple
from keras.utils import register_keras_serializable

def read_images(path: str, image_size: int, num_items: int) -> np.ndarray:
  with gzip.open(path, 'rb') as file:
    data = np.frombuffer(file.read(), np.uint8, offset=16)
    data = data.reshape(num_items, image_size, image_size)
  return data

def read_labels(path: str, num_items: int) -> np.ndarray:
  with gzip.open(path, 'rb') as file:
    data = np.frombuffer(file.read(num_items + 8), np.uint8, offset=8)
    data = data.astype(np.int64)
  return data

def get_data(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    image_size = 28

    # Use TensorFlow's built-in loader (automatically downloads data if missing)
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    training_images = training_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Shuffle + batch
    train_dataset = train_dataset.shuffle(500).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return (train_dataset, test_dataset)


@register_keras_serializable()
class NeuralNetwork(tf.keras.Model):
  def __init__(self, trainable=True, dtype=None, **kwargs):
    super(NeuralNetwork, self).__init__()
    self.sequence = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(20, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime

  
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