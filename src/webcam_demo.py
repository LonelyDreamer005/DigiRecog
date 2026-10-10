"""
webcam_demo.py
--------------
Direct Laptop Webcam Handwritten Digit Recognition using OpenCV & Softmax Regression.

Features:
- Live video feed via cv2.VideoCapture(0)
- Bounding box (Region of Interest - ROI) guide for positioning paper
- Real-time preprocessing and 28x28 MNIST binary canvas preview
- Top predicted digit & Softmax confidence score display
- Live 10-class probability bar display on HUD overlay
- Controls:
    'q' or 'ESC' : Exit webcam window
    's'         : Save current frame & prediction to outputs/
    'c'         : Toggle continuous real-time prediction vs manual freeze
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_digit_roi


def draw_probability_hud(frame, probabilities, predicted_class, confidence):
    """
    Draws a sleek side panel HUD overlay showing all 10 digit probabilities.
    """
    h, w, _ = frame.shape
    panel_w = 220
    panel_x = w - panel_w - 10
    panel_y = 10
    panel_h = 320

    # Semi-transparent dark background for HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

    # Title
    cv2.putText(frame, "Probabilities", (panel_x + 15, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    bar_max_w = 110
    for digit in range(10):
        prob = probabilities[digit]
        y_pos = panel_y + 55 + digit * 25

        # Class label
        color = (0, 255, 120) if digit == predicted_class else (200, 200, 200)
        cv2.putText(frame, f"{digit}:", (panel_x + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Bar
        bar_w = int(prob * bar_max_w)
        bar_color = (0, 230, 115) if digit == predicted_class else (180, 120, 50)
        cv2.rectangle(frame, (panel_x + 40, y_pos - 10), (panel_x + 40 + bar_w, y_pos), bar_color, -1)
        cv2.rectangle(frame, (panel_x + 40, y_pos - 10), (panel_x + 40 + bar_max_w, y_pos), (100, 100, 100), 1)

        # Percentage text
        cv2.putText(frame, f"{prob * 100:4.1f}%", (panel_x + 45 + bar_max_w, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def run_webcam_app():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "digit_softmax.keras")

    if not os.path.exists(model_path):
        print(f"[!] Model not found at: {model_path}")
        print("    Please run: python src/train_softmax.py first.")
        return

    print("[*] Loading Softmax model...")
    model = tf.keras.models.load_model(model_path)

    print("[*] Initializing Laptop Webcam (Index 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Error: Could not open webcam. Make sure no other application is using it.")
        return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ROI (Region of Interest) box parameters (Centered square)
    box_size = 200
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1 = (frame_w - box_size) // 2
    y1 = (frame_h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    print("\n" + "=" * 60)
    print(" WEBCAM DIGIT RECOGNITION ACTIVE")
    print(" Hold a handwritten digit on white paper inside the green box.")
    print(" Keys:")
    print("   [q] or [ESC] : Quit")
    print("   [s]          : Save snapshot")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from camera.")
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        # Preprocess ROI for MNIST model
        canvas_28, input_vector, debug_info = preprocess_digit_roi(roi)

        if input_vector is not None:
            # Predict
            predictions = model.predict(input_vector, verbose=0)[0]
            pred_digit = int(np.argmax(predictions))
            confidence = float(predictions[pred_digit] * 100.0)

            # Draw green ROI bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Place Digit Here", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # Display prediction banner at top left
            conf_color = (0, 255, 0) if confidence > 70 else ((0, 220, 255) if confidence > 45 else (0, 0, 255))
            cv2.rectangle(frame, (10, 10), (280, 85), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (280, 85), conf_color, 2)
            cv2.putText(frame, f"Prediction: {pred_digit}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, conf_color, 2, cv2.LINE_AA)

            # Draw 10-class probability side HUD
            draw_probability_hud(frame, predictions, pred_digit, confidence)

            # Overlay 28x28 processed binary view in bottom left
            preview_scaled = cv2.resize(canvas_28, (84, 84), interpolation=cv2.INTER_NEAREST)
            preview_bgr = cv2.cvtColor(preview_scaled, cv2.COLOR_GRAY2BGR)
            # Add border
            cv2.rectangle(preview_bgr, (0, 0), (83, 83), (0, 255, 0), 1)
            frame[frame_h - 94 : frame_h - 10, 10 : 94] = preview_bgr
            cv2.putText(frame, "Model Input (28x28)", (10, frame_h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Smart Digit Recognition - Softmax Baseline", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC
            break
        elif key == ord('s'):
            snapshot_path = os.path.join(project_root, "outputs", "webcam_snapshot.png")
            cv2.imwrite(snapshot_path, frame)
            print(f"[+] Saved snapshot to: {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Webcam session ended.")


if __name__ == "__main__":
    run_webcam_app()
