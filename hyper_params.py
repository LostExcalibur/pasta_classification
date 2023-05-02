# %%
import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras_tuner import RandomSearch
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models, optimizers

warnings.filterwarnings("ignore")


# %%
def load_image(path: str):
    return np.array(copy.deepcopy(Image.open(path).convert("RGB")))


# %%
labels = os.listdir("dataset/")


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

print(X_train.shape)

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
# model = models.Sequential()
# model.add(layers.Conv2D(filters=, activation='relu', input_shape=(256, 256, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(7))


def build_model(hp):
    print("Building model")
    # create model object
    model = models.Sequential(
        [
            # adding first convolutional layer
            layers.Conv2D(
                # adding filter
                filters=hp.Int("conv_1_filter", min_value=32, max_value=128, step=16),
                # adding filter size or kernel size
                kernel_size=hp.Int("conv_1_kernel", min_value=2, max_value=8, step=2),
                # activation function
                activation="relu",
                input_shape=(256, 256, 3),
            ),
            # adding second convolutional layer
            layers.MaxPooling2D(
                pool_size=hp.Int("max_pool_1", min_value=2, max_value=8, step=2)
            ),
            layers.Conv2D(
                # adding filter
                filters=hp.Int("conv_2_filter", min_value=32, max_value=128, step=16),
                # adding filter size or kernel size
                kernel_size=hp.Int("conv_2_kernel", min_value=2, max_value=8, step=2),
                # activation function
                activation="relu",
            ),
            layers.MaxPooling2D(
                pool_size=hp.Int("max_pool_2", min_value=2, max_value=8, step=2)
            ),
            # adding flatten layer
            layers.Flatten(),
            # adding dense layer
            layers.Dense(
                64,
                activation="relu",
            ),
            # output layer
            layers.Dense(7, activation="softmax"),
        ]
    )
    model.summary()

    # compilation of model
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # 'sparse_categorical_crossentropy'
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# %%
tuner = RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True
)
tuner.search_space_summary()

# %%
# history = model.fit(X_train, Y_train, epochs=10,
#                     validation_data=(X_test, Y_test))
tuner.search(X_train, Y_train, epochs=10, validation_data=(X_train, Y_train))

model = tuner.get_best_models(num_models=1)[0]
model.summary()

# %%
history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))

# %%
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

# %%
if not os.path.exists("models"):
    os.mkdir("models")

model.save("models/hyper_params")
