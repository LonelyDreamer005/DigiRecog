"""
evaluate_misclassifications.py
------------------------------
Error Analysis & Misclassified Samples Visualizer for Softmax Regression.

Identifies, extracts, and plots real test examples that the Softmax Regression
model misclassified, displaying:
- True Label vs. Predicted Label
- Prediction Confidence (%)
- Visual explanation of why the linear classifier failed

Outputs:
- outputs/misclassified_samples.png
- outputs/misclassified_summary.json
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def analyze_misclassifications(max_display=15):
    model_path = os.path.join(PROJECT_ROOT, "models", "digit_softmax.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_softmax.py first.")

    print("[1/3] Loading MNIST test dataset & Softmax model...")
    (_, _), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
    x_test_norm = x_test_raw.astype("float32") / 255.0
    x_test_flat = x_test_norm.reshape((-1, 784))

    model = tf.keras.models.load_model(model_path)

    print("[2/3] Computing predictions and filtering misclassifications...")
    probabilities = model.predict(x_test_flat, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    misclassified_indices = np.where(predictions != y_test)[0]
    total_errors = len(misclassified_indices)
    total_test = len(y_test)
    error_rate = (total_errors / total_test) * 100

    print(f"      Total Test Samples:    {total_test:,}")
    print(f"      Misclassified Samples: {total_errors:,} ({error_rate:.2f}% error rate)")

    # Select top 15 misclassified samples with highest confidence (most severe errors)
    confidences = np.max(probabilities, axis=1)
    error_confidences = confidences[misclassified_indices]
    sorted_error_order = np.argsort(-error_confidences)  # Descending order of confidence
    top_misclassified_idx = misclassified_indices[sorted_order[:max_display]] if 'sorted_order' in locals() else misclassified_indices[sorted_error_order[:max_display]]

    # Plot misclassified grid
    print(f"[3/3] Generating misclassified samples plot ({max_display} examples)...")
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    fig.suptitle(f"Softmax Regression: Analysis of Misclassified Test Samples (Total Errors: {total_errors}/10,000)",
                 fontsize=15, fontweight="bold", y=0.98)

    misclassified_records = []

    for i, idx in enumerate(top_misclassified_idx):
        ax = axes[i // cols, i % cols]
        img = x_test_raw[idx]
        true_lbl = int(y_test[idx])
        pred_lbl = int(predictions[idx])
        conf = float(confidences[idx] * 100)
        true_lbl_conf = float(probabilities[idx][true_lbl] * 100)

        record = {
            "test_index": int(idx),
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "predicted_confidence_pct": round(conf, 2),
            "true_label_probability_pct": round(true_lbl_conf, 2)
        }
        misclassified_records.append(record)

        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}\nConf: {conf:.1f}% (P({true_lbl})={true_lbl_conf:.1f}%)",
                     fontsize=10, fontweight="bold",
                     color="#dc2626" if conf > 70 else "#ea580c")
        ax.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(PROJECT_ROOT, "outputs", "misclassified_samples.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved misclassified samples plot to: {plot_path}")

    # Save summary JSON
    json_path = os.path.join(PROJECT_ROOT, "outputs", "misclassified_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "total_test_samples": total_test,
            "total_misclassified": total_errors,
            "error_rate_pct": round(error_rate, 2),
            "top_high_confidence_errors": misclassified_records
        }, f, indent=4)
    print(f"[+] Saved misclassified summary to: {json_path}")

    return total_errors, misclassified_records


if __name__ == "__main__":
    analyze_misclassifications()
