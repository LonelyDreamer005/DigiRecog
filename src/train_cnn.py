"""
train_cnn.py
------------
Convolutional Neural Network (CNN) Deep Learning Extension for MNIST.

Architecture:
- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Flatten
- Dropout (0.25)
- Dense (128, ReLU)
- Dense (10, Softmax)

Objective:
- Train a spatial deep-learning model to compare with the linear Softmax Regression baseline.
- Save trained CNN to `models/digit_cnn.keras`.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_mnist_for_cnn():
    print("[1/3] Loading MNIST dataset for CNN (28x28x1)...")
    (x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to (N, 28, 28, 1) and normalize to [0, 1]
    x_train = x_train_raw.astype("float32") / 255.0
    x_test = x_test_raw.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def build_cnn_model():
    print("[2/3] Building Convolutional Neural Network (CNN)...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


def train_and_save_cnn():
    (x_train, y_train), (x_test, y_test) = load_mnist_for_cnn()
    model = build_cnn_model()

    print("[3/3] Training CNN model (5 epochs, batch_size=128)...")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 60)
    print(f" FINAL TEST RESULTS (CNN Deep Learning Model)")
    print(f" Test Loss:     {test_loss:.4f}")
    print(f" Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_path = os.path.join(project_root, "models", "digit_cnn.keras")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[+] CNN successfully saved to: {model_save_path}")

    return test_loss, test_accuracy


if __name__ == "__main__":
    train_and_save_cnn()
