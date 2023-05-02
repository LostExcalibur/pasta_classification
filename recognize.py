import tensorflow as tf
import os
import numpy as np
from PIL import Image


labels = os.listdir("dataset/")

model = tf.saved_model.load("models/first_try")

image = np.array(Image.open("Image_test_à_définir.png"))

print(labels[model.predict(image)])