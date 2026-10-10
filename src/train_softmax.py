"""
train_softmax.py
----------------
Softmax Regression Baseline for Handwritten Digit Recognition on MNIST.

Architecture:
- Input: 784 features (flattened 28x28 grayscale image)
- Output Layer: 10 neurons with Softmax activation
- Total Parameters: 784 * 10 (weights) + 10 (biases) = 7,850

Objective:
- Train the linear baseline using SGD optimizer and sparse categorical cross-entropy.
- Save the trained model to `models/digit_softmax.keras`.
- Plot and save training/validation loss and accuracy curves to `outputs/training_curves.png`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_and_preprocess_mnist():
    """
    Loads MNIST dataset and preprocesses:
    - Normalization: Scales pixel intensities from [0, 255] to [0.0, 1.0].
    - Reshaping: Flattens 28x28 2D arrays into 784 1D feature vectors.
    """
    print("[1/4] Loading MNIST dataset...")
    (x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] range
    x_train = x_train_raw.astype("float32") / 255.0
    x_test = x_test_raw.astype("float32") / 255.0

    # Flatten 28x28 images into 784-dimensional vectors
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))

    print(f"      Training set: {x_train.shape[0]} samples, shape: {x_train.shape}")
    print(f"      Test set:     {x_test.shape[0]} samples, shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def build_softmax_model():
    """
    Constructs the single-layer Softmax Regression model.
    """
    print("[2/4] Building Softmax Regression model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,), name="input_pixels"),
        tf.keras.layers.Dense(10, activation="softmax", name="softmax_output")
    ])

    # Compile with SGD optimizer (lr=0.1) and Sparse Categorical Crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


def plot_training_history(history, save_path):
    """
    Plots training & validation accuracy and loss curves and saves the figure.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#2563eb")
    plt.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#16a34a")
    plt.title("Model Accuracy over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.85, 1.0])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, "o-", label="Training Loss", color="#dc2626")
    plt.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#ea580c")
    plt.title("Model Loss over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved training graphs to: {save_path}")


def train_and_evaluate():
    # 1. Prepare data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    # 2. Build model
    model = build_softmax_model()

    # 3. Train model
    print("[3/4] Training Softmax Regression model (10 epochs, batch_size=128)...")
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # 4. Evaluate on test set
    print("[4/4] Evaluating on 10,000 test images...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 60)
    print(f" FINAL TEST RESULTS (Softmax Regression Baseline)")
    print(f" Test Loss:     {test_loss:.4f}")
    print(f" Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 60)

    # Ensure output directories exist
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_save_path = os.path.join(project_root, "models", "digit_softmax.keras")
    graph_save_path = os.path.join(project_root, "outputs", "training_curves.png")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[+] Model successfully saved to: {model_save_path}")

    # Plot and save curves
    plot_training_history(history, graph_save_path)

    return test_loss, test_accuracy


if __name__ == "__main__":
    train_and_evaluate()
