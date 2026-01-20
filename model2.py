import tensorflow as tf
from tensorflow.keras import models, layers #type: ignore
from tensorflow.keras.applications import VGG16 #type: ignore
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.array([tf.image.resize(img[..., np.newaxis], (32, 32)).numpy() for img in x_train])
x_test = np.array([tf.image.resize(img[..., np.newaxis], (32, 32)).numpy() for img in x_test])
x_train = np.repeat(x_train, 3, axis=-1).astype(np.float32) / 255.
x_test = np.repeat(x_test, 3, axis=-1).astype(np.float32) / 255.

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

base_model.trainable = False
model = models.Sequential([
    base_model,

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train, y_train,
    batch_size=256,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

model.save('model2.keras')