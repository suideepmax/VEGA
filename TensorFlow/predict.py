import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import matplotlib.pyplot as plt

from kintro import *

model = tf.keras.models.load_model('outputs/model.keras') # Load the trained model

url = 'https://raw.githubusercontent.com/MicrosoftDocs/tensorflow-learning-path/main/intro-keras/predict-image.png' # URL of the image to be predicted
with Image.open(requests.get(url, stream=True).raw) as image: # Open the image from the URL
  X = np.asarray(image, dtype=np.float32).reshape((-1, 28, 28)) / 255.0 # Preprocess the image

plt.figure()
plt.axis('off')
plt.imshow(X.squeeze(), cmap='gray')
plt.show()

predicted_vector = model.predict(X) # Predict the class of the image
# how do you predict the class from the predicted vector?
# the predicted vector is a vector of 10 values, one for each class
# the class with the highest value is the predicted class
predicted_index = np.argmax(predicted_vector) # Get the index of the class with the highest value
predicted_name = labels_map[predicted_index] # Get the name of the predicted class

# what do the above three lines do?
# 1. Get the predicted vector from the model
# 2. Find the index of the class with the highest predicted value
# 3. Map the index to the class name using the labels_map
# this is done because the model outputs a vector of probabilities for each class
print(f'Predicted class: {predicted_name}')