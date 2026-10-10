"""
compare_models.py
-----------------
Comprehensive Comparison: Softmax Regression (Linear ML) vs. CNN (Deep Learning).

Evaluates:
1. Classification Accuracy (MNIST 10k Test Set)
2. Model Complexity & Parameter Count
3. Inference Latency (ms per image)
4. Robustness comparison under distortions (N=2000)
5. Confusion Matrices for class-by-class error analysis

Outputs:
- outputs/model_comparison.png
- outputs/confusion_matrices.png
- outputs/comparison_summary.json
"""

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

from src.robustness import (
    load_subset,
    apply_gaussian_noise,
    apply_translation,
    apply_rotation,
    apply_gaussian_blur
)


def run_full_comparison():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    softmax_path = os.path.join(project_root, "models", "digit_softmax.keras")
    cnn_path = os.path.join(project_root, "models", "digit_cnn.keras")

    if not os.path.exists(softmax_path) or not os.path.exists(cnn_path):
        raise FileNotFoundError("Both digit_softmax.keras and digit_cnn.keras must exist.")

    print("[1/5] Loading models...")
    softmax_model = tf.keras.models.load_model(softmax_path)
    cnn_model = tf.keras.models.load_model(cnn_path)

    # 1. Parameter counts
    softmax_params = softmax_model.count_params()
    cnn_params = cnn_model.count_params()

    # 2. Load 10k test data
    print("[2/5] Evaluating test accuracy on full 10,000 MNIST test images...")
    (_, _), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
    x_test_norm = x_test_raw.astype("float32") / 255.0

    x_test_flat = x_test_norm.reshape((-1, 784))
    x_test_cnn = np.expand_dims(x_test_norm, -1)

    _, softmax_test_acc = softmax_model.evaluate(x_test_flat, y_test, verbose=0)
    _, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)

    # 3. Measure inference speed (latency per 1,000 predictions)
    print("[3/5] Measuring inference latency...")
    warmup_iters = 50
    test_batch_flat = x_test_flat[:500]
    test_batch_cnn = x_test_cnn[:500]

    # Warmup
    _ = softmax_model.predict(test_batch_flat, verbose=0)
    _ = cnn_model.predict(test_batch_cnn, verbose=0)

    # Softmax timing
    t0 = time.perf_counter()
    for _ in range(10):
        _ = softmax_model.predict(test_batch_flat, verbose=0)
    softmax_latency_ms = ((time.perf_counter() - t0) / (10 * 500)) * 1000.0

    # CNN timing
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cnn_model.predict(test_batch_cnn, verbose=0)
    cnn_latency_ms = ((time.perf_counter() - t0) / (10 * 500)) * 1000.0

    # 4. Compare Robustness on 2000 sampled images
    print("[4/5] Comparing robustness across 5 distortion conditions (N=2000)...")
    x_sample_2d, y_sample = load_subset(sample_size=2000, seed=42)
    conditions = {
        "Original": x_sample_2d,
        "Noise": apply_gaussian_noise(x_sample_2d, sigma=0.25),
        "Shift (2px)": apply_translation(x_sample_2d, shift_pixels=2),
        "Rotation (15°)": apply_rotation(x_sample_2d, angle=15),
        "Blur": apply_gaussian_blur(x_sample_2d, ksize=(3, 3), sigma=1.0)
    }

    robustness_softmax = {}
    robustness_cnn = {}

    for name, img_set in conditions.items():
        # Softmax evaluation
        flat = img_set.reshape((-1, 784))
        _, acc_s = softmax_model.evaluate(flat, y_sample, verbose=0)
        robustness_softmax[name] = round(float(acc_s * 100), 2)

        # CNN evaluation
        cnn_in = np.expand_dims(img_set, -1)
        _, acc_c = cnn_model.evaluate(cnn_in, y_sample, verbose=0)
        robustness_cnn[name] = round(float(acc_c * 100), 2)

    # 5. Confusion Matrices
    print("[5/5] Generating confusion matrices & saving plots...")
    y_pred_softmax = np.argmax(softmax_model.predict(x_test_flat, verbose=0), axis=1)
    y_pred_cnn = np.argmax(cnn_model.predict(x_test_cnn, verbose=0), axis=1)

    cm_softmax = confusion_matrix(y_test, y_pred_softmax)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm_softmax).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title(f"Softmax Regression Confusion Matrix\n(Accuracy: {softmax_test_acc*100:.2f}%)",
                      fontsize=12, fontweight="bold")

    ConfusionMatrixDisplay(confusion_matrix=cm_cnn).plot(
        ax=axes[1], cmap="Greens", colorbar=False
    )
    axes[1].set_title(f"CNN Model Confusion Matrix\n(Accuracy: {cnn_test_acc*100:.2f}%)",
                      fontsize=12, fontweight="bold")

    plt.tight_layout()
    cm_save_path = os.path.join(project_root, "outputs", "confusion_matrices.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices to: {cm_save_path}")

    # Plot Robustness Comparison Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(11, 6))
    cond_keys = list(conditions.keys())
    x_indices = np.arange(len(cond_keys))
    width = 0.35

    bars1 = ax.bar(x_indices - width/2, [robustness_softmax[k] for k in cond_keys],
                   width, label=f"Softmax Baseline ({softmax_params:,} params)", color="#3b82f6")
    bars2 = ax.bar(x_indices + width/2, [robustness_cnn[k] for k in cond_keys],
                   width, label=f"CNN Extension ({cnn_params:,} params)", color="#10b981")

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison: Robustness under Distortions (Softmax vs CNN)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(cond_keys, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=11)

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    comp_chart_path = os.path.join(project_root, "outputs", "model_comparison.png")
    plt.tight_layout()
    plt.savefig(comp_chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparison chart to: {comp_chart_path}")

    # Save summary dictionary
    summary = {
        "Softmax Regression (Baseline)": {
            "parameters": int(softmax_params),
            "test_accuracy_pct": round(float(softmax_test_acc * 100), 2),
            "latency_ms_per_image": round(float(softmax_latency_ms), 3),
            "robustness": robustness_softmax
        },
        "CNN Deep Learning (Extension)": {
            "parameters": int(cnn_params),
            "test_accuracy_pct": round(float(cnn_test_acc * 100), 2),
            "latency_ms_per_image": round(float(cnn_latency_ms), 3),
            "robustness": robustness_cnn
        }
    }

    summary_path = os.path.join(project_root, "outputs", "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[+] Saved comparison summary to: {summary_path}")

    print("\n" + "=" * 70)
    print(" SUMMARY COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Softmax Regression':<20} | {'CNN Deep Learning':<20}")
    print("-" * 70)
    print(f"{'Parameters':<25} | {softmax_params:<20,d} | {cnn_params:<20,d}")
    print(f"{'Clean Test Accuracy':<25} | {softmax_test_acc*100:<19.2f}% | {cnn_test_acc*100:<19.2f}%")
    print(f"{'Inference Latency/Img':<25} | {softmax_latency_ms:<17.3f} ms | {cnn_latency_ms:<17.3f} ms")
    print(f"{'Shift (2px) Robustness':<25} | {robustness_softmax['Shift (2px)']:<19.2f}% | {robustness_cnn['Shift (2px)']:<19.2f}%")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_full_comparison()
