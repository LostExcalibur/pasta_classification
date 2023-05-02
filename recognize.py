import copy
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(path: str):
    return np.array(Image.open(path).convert("RGB"))


labels = sorted(os.listdir("dataset/"))

model: tf.keras.models.Sequential = tf.keras.models.load_model("models/first_try")

model.summary()

image = load_image("alois.png").reshape(1, 256, 256, 3)

print(labels[np.argmax(model.predict(image))])
