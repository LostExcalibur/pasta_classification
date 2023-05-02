# %%
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models


# %%
def load_image(path: str):
    return np.array(copy.deepcopy(Image.open(path).convert("RGB")))


# %%
labels = os.listdir("dataset/")
print(labels)


# %%
def prepare_dataset():
    X, Y = [], []

    for i, label in enumerate(labels):
        image_names = os.listdir("dataset/" + label)
        paths = list(map(lambda file: f"dataset/{label}/{file}", image_names))

        num = len(paths)

        X.extend(map(load_image, paths))
        Y.extend([i] * num)

    return train_test_split(np.array(X), np.array(Y), shuffle=True)


# %%
X_train, X_test, Y_train, Y_test = prepare_dataset()

# %%
plt.figure(figsize=[5, 5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(X_train[0])
plt.title("Ground Truth : {}".format(labels[Y_train[0]]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(X_test[0])
plt.title("Ground Truth : {}".format(labels[Y_test[0]]))

plt.show()

# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(7, activation="softmax"))

model.summary()

# %%
model.compile(
    optimizer="adam",
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# %%
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# %%
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

# %%

if not os.path.exists("models"):
    os.mkdir("models")

model.save("models/first_try")
