"""
robustness.py
-------------
Robustness Analysis of Softmax Regression Classifier on MNIST.

Evaluates model performance under real-world visual distortions:
1. Original clean test images
2. Gaussian Noise (simulates sensor noise)
3. 2-Pixel Shift/Translation (simulates imperfect centering)
4. 15-Degree Rotation (simulates hand tilt)
5. Gaussian Blur (simulates out-of-focus camera capture)

Samples: 2,000 test images randomly chosen with fixed seed (42).
Outputs:
- outputs/robustness_comparison.png (Bar chart with value annotations)
- outputs/robustness_results.json (Exact numerical data for reports)
"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_subset(sample_size=2000, seed=42):
    """
    Loads MNIST test data and samples a reproducible subset of 28x28 images.
    """
    print(f"[1/4] Loading MNIST test set and sampling {sample_size} images (seed={seed})...")
    (_, _), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

    np.random.seed(seed)
    indices = np.random.choice(len(x_test_raw), size=sample_size, replace=False)

    # Normalize to [0.0, 1.0] and keep 28x28 shape for spatial transforms
    x_sample = x_test_raw[indices].astype("float32") / 255.0
    y_sample = y_test_raw[indices]

    return x_sample, y_sample


def apply_gaussian_noise(images, mean=0.0, sigma=0.25):
    """
    Adds zero-mean Gaussian noise to images and clips values to [0.0, 1.0].
    """
    noisy_images = []
    for img in images:
        noise = np.random.normal(mean, sigma, img.shape)
        noisy_img = np.clip(img + noise, 0.0, 1.0).astype("float32")
        noisy_images.append(noisy_img)
    return np.array(noisy_images)


def apply_translation(images, shift_pixels=2):
    """
    Shifts images by shift_pixels along x and y directions using affine transformation.
    """
    shifted_images = []
    # Translation matrix for (+2, +2) shift
    M = np.float32([[1, 0, shift_pixels], [0, 1, shift_pixels]])
    for img in images:
        shifted = cv2.warpAffine(img, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shifted_images.append(shifted.astype("float32"))
    return np.array(shifted_images)


def apply_rotation(images, angle=15):
    """
    Rotates images around their center (14, 14) by the specified angle in degrees.
    """
    rotated_images = []
    center = (14, 14)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    for img in images:
        rotated = cv2.warpAffine(img, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_images.append(rotated.astype("float32"))
    return np.array(rotated_images)


def apply_gaussian_blur(images, ksize=(3, 3), sigma=1.0):
    """
    Applies Gaussian blur with a specified kernel size and standard deviation.
    """
    blurred_images = []
    for img in images:
        blurred = cv2.GaussianBlur(img, ksize, sigmaX=sigma)
        blurred_images.append(blurred.astype("float32"))
    return np.array(blurred_images)


def run_robustness_experiment():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "digit_softmax.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train_softmax.py first.")

    print(f"[2/4] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    x_sample_2d, y_sample = load_subset(sample_size=2000, seed=42)

    print("[3/4] Generating distorted test conditions...")
    conditions = {
        "Original": x_sample_2d,
        "Gaussian Noise": apply_gaussian_noise(x_sample_2d, sigma=0.25),
        "Shift (2px)": apply_translation(x_sample_2d, shift_pixels=2),
        "Rotation (15°)": apply_rotation(x_sample_2d, angle=15),
        "Gaussian Blur": apply_gaussian_blur(x_sample_2d, ksize=(3, 3), sigma=1.0)
    }

    results = {}
    print("\n" + "=" * 65)
    print(" ROBUSTNESS EVALUATION (2,000 Test Images)")
    print("=" * 65)

    for condition_name, img_set in conditions.items():
        # Flatten from (2000, 28, 28) to (2000, 784) for Softmax input
        flattened = img_set.reshape((-1, 784))
        _, acc = model.evaluate(flattened, y_sample, verbose=0)
        acc_pct = float(acc * 100)
        results[condition_name] = round(acc_pct, 2)
        print(f" • {condition_name:<18} Accuracy: {acc_pct:6.2f}%")

    print("=" * 65)

    # Save numerical results to JSON
    json_path = os.path.join(project_root, "outputs", "robustness_results.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Saved numerical results to: {json_path}")

    # Plot Bar Chart
    print("[4/4] Plotting robustness comparison bar chart...")
    names = list(results.keys())
    accuracies = list(results.values())
    colors = ["#2563eb", "#ef4444", "#f59e0b", "#8b5cf6", "#10b981"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    plt.title("Softmax Regression Robustness to Input Distortions\n(N=2,000 MNIST Test Samples)",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Distortion Condition", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Classification Accuracy (%)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.0,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    chart_path = os.path.join(project_root, "outputs", "robustness_comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparison chart to: {chart_path}\n")

    return results


if __name__ == "__main__":
    run_robustness_experiment()
