import tensorflow as tf
import os
import numpy as np
from PIL import Image
import copy

def load_image(path: str):
    return np.array(Image.open(path).convert("RGB"))

labels = os.listdir("dataset/")

model = tf.keras.models.load_model("models/first_try")

model.summary()

image = load_image("dataset/penne/chicken-penne-pasta_resized.png").reshape(1, 256, 256, 3)


print(model.predict(image))