import copy
import os

from sys import argv

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(path: str):
    return np.array(Image.open(path).convert("RGB"))


labels = sorted(os.listdir("dataset/"))

model: tf.keras.models.Sequential = tf.keras.models.load_model("models/first_try")

model.summary()

filename = argv[1] if len(argv) > 1 else "alois.png"
image = load_image(filename).reshape(1, 256, 256, 3)

print(f"{filename} has been recognized as a {labels[np.argmax(model.predict(image))]}")
