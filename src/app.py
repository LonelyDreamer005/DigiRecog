"""
app.py
------
Flask Web Application for Smart Handwritten Digit Recognition.

Features:
- Dual Model Support: Softmax Regression (Linear Baseline) & CNN (Deep Learning Extension)
- Side-by-Side Model Comparison mode
- Multi-input: Interactive HTML5 Canvas drawing, Laptop Webcam snapshot, and Image upload
- Live 10-class probability distribution visualization
- 28x28 preprocessed input viewer
- Embedded experimental results (Training curves, Robustness bar charts, Confusion matrices)
"""

import base64
import io
import json
import os
import sys
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_digit_roi
import tensorflow as tf

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)

# Load Trained Models into Memory
print("[*] Initializing models for Flask server...")
SOFTMAX_PATH = os.path.join(PROJECT_ROOT, "models", "digit_softmax.keras")
CNN_PATH = os.path.join(PROJECT_ROOT, "models", "digit_cnn.keras")

softmax_model = None
cnn_model = None

if os.path.exists(SOFTMAX_PATH):
    softmax_model = tf.keras.models.load_model(SOFTMAX_PATH)
    print(" [+] Softmax model loaded successfully.")
else:
    print(f" [!] Warning: Softmax model not found at {SOFTMAX_PATH}")

if os.path.exists(CNN_PATH):
    cnn_model = tf.keras.models.load_model(CNN_PATH)
    print(" [+] CNN model loaded successfully.")
else:
    print(f" [!] Warning: CNN model not found at {CNN_PATH}")


def decode_base64_image(base64_str):
    """
    Decodes base64 data URI to an OpenCV BGR numpy array.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def encode_canvas_to_base64(canvas_28x28):
    """
    Encodes 28x28 uint8 array to a base64 PNG data URL for browser display.
    """
    # Scale up slightly for crisp web rendering
    scaled = cv2.resize(canvas_28x28, (112, 112), interpolation=cv2.INTER_NEAREST)
    _, buffer = cv2.imencode(".png", scaled)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """
    Serves generated experiment charts from the outputs/ directory.
    """
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    return send_from_directory(outputs_dir, filename)


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """
    Returns actual measured evaluation & comparison metrics.
    """
    summary_path = os.path.join(PROJECT_ROOT, "outputs", "comparison_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return jsonify({"status": "success", "data": data})
    return jsonify({"status": "error", "message": "Summary not yet generated"}), 404


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    image_b64 = data.get("image")
    model_choice = data.get("model", "both")  # "softmax", "cnn", or "both"

    if not image_b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # 1. Decode image
        bgr_img = decode_base64_image(image_b64)

        # 2. Preprocess to MNIST standard
        canvas_28, norm_vec_784, debug_info = preprocess_digit_roi(bgr_img)
        if canvas_28 is None or norm_vec_784 is None:
            return jsonify({"error": "Could not preprocess image"}), 400

        preview_b64 = encode_canvas_to_base64(canvas_28)

        # Prepare CNN input (1, 28, 28, 1)
        cnn_input = np.expand_dims(canvas_28.astype("float32") / 255.0, axis=(0, -1))

        results = {
            "preprocessed_image": preview_b64,
            "predictions": {}
        }

        # Softmax Prediction
        if softmax_model and model_choice in ["softmax", "both"]:
            s_probs = softmax_model.predict(norm_vec_784, verbose=0)[0]
            s_digit = int(np.argmax(s_probs))
            results["predictions"]["softmax"] = {
                "name": "Softmax Regression (Baseline)",
                "predicted_digit": s_digit,
                "confidence": round(float(s_probs[s_digit] * 100), 2),
                "probabilities": [round(float(p * 100), 2) for p in s_probs],
                "parameters": 7850
            }

        # CNN Prediction
        if cnn_model and model_choice in ["cnn", "both"]:
            c_probs = cnn_model.predict(cnn_input, verbose=0)[0]
            c_digit = int(np.argmax(c_probs))
            results["predictions"]["cnn"] = {
                "name": "CNN (Deep Learning Extension)",
                "predicted_digit": c_digit,
                "confidence": round(float(c_probs[c_digit] * 100), 2),
                "probabilities": [round(float(p * 100), 2) for p in c_probs],
                "parameters": 225034
            }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" FLASK WEB APPLICATION STARTING")
    print(" Open in browser: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
