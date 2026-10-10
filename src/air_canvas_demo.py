"""
air_canvas_demo.py
------------------
Interactive 5-Finger Gesture Air-Drawing with Dual Model Recognition (Softmax + CNN).

Gestures:
- ☝️ Index Finger ONLY UP:       [DRAW MODE]     - Draw continuous strokes in the air
- ✌️ Index + Middle UP:          [HOVER MODE]    - Lift pen / move hand without drawing
- ✋ All 5 Fingers OPEN:         [CLEAR CANVAS]  - Erase the air drawing
- ✊ Fist (0 Fingers UP):        [PREDICT DIGIT] - Run dual prediction (Softmax & CNN)

On-Screen UI Buttons (Hover with index fingertip to click):
- [CLEAR]  - Erases drawing
- [PREDICT]- Runs inference
- [QUIT]   - Closes window
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_digit_roi


def draw_hud(frame, s_digit, s_conf, c_digit, c_conf, probabilities, status_msg, gesture_name, gesture_color):
    h, w, _ = frame.shape

    # 1. Top Gesture & Status Header
    cv2.rectangle(frame, (10, 10), (480, 80), (15, 15, 20), -1)
    cv2.rectangle(frame, (10, 10), (480, 80), gesture_color, 2)
    cv2.putText(frame, f"Gesture: {gesture_name}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, gesture_color, 2, cv2.LINE_AA)
    cv2.putText(frame, status_msg, (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # 2. Virtual On-Screen Buttons (Top Center/Right)
    buttons = [
        ("CLEAR", 510, 10, 630, 65, (0, 0, 180)),
        ("PREDICT", 650, 10, 790, 65, (0, 150, 0)),
        ("QUIT", 810, 10, 910, 65, (50, 50, 60))
    ]
    for text, bx1, by1, bx2, by2, bcolor in buttons:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bcolor, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(frame, text, (bx1 + 15, by1 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Dual Model Prediction Box (Top Left below header)
    if s_digit is not None:
        cv2.rectangle(frame, (10, 90), (320, 200), (15, 15, 20), -1)
        cv2.rectangle(frame, (10, 90), (320, 200), (0, 255, 120), 2)

        # Softmax baseline
        cv2.putText(frame, f"Softmax (Baseline):  {s_digit} ({s_conf:.1f}%)", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 200, 255), 2, cv2.LINE_AA)

        # CNN deep model
        if c_digit is not None:
            cv2.putText(frame, f"CNN (Deep Model):    {c_digit} ({c_conf:.1f}%)", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 2, cv2.LINE_AA)

    # 4. Right Side 10-Class Probability Bars HUD
    panel_w = 200
    panel_x = w - panel_w - 10
    panel_y = 10
    panel_h = 320

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (70, 70, 80), 1)

    cv2.putText(frame, "Probabilities (0-9)", (panel_x + 10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    bar_max_w = 95
    for digit in range(10):
        prob = probabilities[digit]
        y_pos = panel_y + 48 + digit * 26

        is_pred = (digit == s_digit)
        color = (0, 255, 120) if is_pred else (190, 190, 190)
        cv2.putText(frame, f"{digit}:", (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        bar_w = int(prob * bar_max_w)
        bar_color = (0, 230, 115) if is_pred else (180, 120, 50)
        cv2.rectangle(frame, (panel_x + 32, y_pos - 9), (panel_x + 32 + bar_w, y_pos), bar_color, -1)
        cv2.rectangle(frame, (panel_x + 32, y_pos - 9), (panel_x + 32 + bar_max_w, y_pos), (70, 70, 70), 1)

        cv2.putText(frame, f"{prob * 100:4.1f}%", (panel_x + 38 + bar_max_w, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)


def run_air_canvas():
    print("=" * 65)
    print(" 🖐️ AIR CANVAS: 5-FINGER GESTURE DIGIT RECOGNITION ACTIVE")
    print("=" * 65)

    # 1. Load Models
    softmax_path = os.path.join(PROJECT_ROOT, "models", "digit_softmax.keras")
    cnn_path = os.path.join(PROJECT_ROOT, "models", "digit_cnn.keras")

    softmax_model = tf.keras.models.load_model(softmax_path) if os.path.exists(softmax_path) else None
    cnn_model = tf.keras.models.load_model(cnn_path) if os.path.exists(cnn_path) else None

    # 2. Camera & Detector Setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(maxHands=1, detectionCon=0.75, minTrackCon=0.75)

    draw_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    prev_x, prev_y = 0, 0
    brush_thickness = 26

    s_digit, s_conf = None, 0.0
    c_digit, c_conf = None, 0.0
    probabilities = np.zeros(10)
    last_canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    status_msg = "Ready: Raise Index Finger to Draw"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if draw_canvas.shape[:2] != (h, w):
            draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        hands, _ = detector.findHands(frame, draw=False)

        current_gesture = "No Hand Detected"
        gesture_color = (120, 120, 120)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)  # [Thumb, Index, Middle, Ring, Pinky]

            x_idx, y_idx = lmList[8][0], lmList[8][1]

            # Check if index finger is hovering over virtual UI buttons
            if y_idx < 70:
                if 510 < x_idx < 630:  # CLEAR
                    draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    s_digit = None
                    probabilities = np.zeros(10)
                    status_msg = "Canvas Cleared via Button"
                    prev_x, prev_y = 0, 0
                elif 650 < x_idx < 790:  # PREDICT
                    prev_x, prev_y = 0, 0
                    gray_c = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                    if np.count_nonzero(gray_c) > 60:
                        c28, vec, _ = preprocess_digit_roi(draw_canvas)
                        if vec is not None:
                            last_canvas_28 = c28
                            s_probs = softmax_model.predict(vec, verbose=0)[0]
                            s_digit = int(np.argmax(s_probs))
                            s_conf = float(s_probs[s_digit] * 100)
                            probabilities = s_probs

                            if cnn_model:
                                c_in = np.expand_dims(c28.astype("float32")/255.0, (0, -1))
                                c_probs = cnn_model.predict(c_in, verbose=0)[0]
                                c_digit = int(np.argmax(c_probs))
                                c_conf = float(c_probs[c_digit] * 100)

                            status_msg = f"Softmax: {s_digit} ({s_conf:.1f}%) | CNN: {c_digit} ({c_conf:.1f}%)"
                elif 810 < x_idx < 910:  # QUIT
                    break

            # GESTURE 1: DRAW MODE (Index only UP: [0, 1, 0, 0, 0])
            elif fingers == [0, 1, 0, 0, 0] or (fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0):
                current_gesture = "DRAWING ✍️"
                gesture_color = (0, 255, 0)
                status_msg = "Drawing in the air..."

                cv2.circle(frame, (x_idx, y_idx), 12, (0, 255, 0), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x_idx, y_idx

                cv2.line(draw_canvas, (prev_x, prev_y), (x_idx, y_idx), (255, 255, 255), brush_thickness)
                prev_x, prev_y = x_idx, y_idx

            # GESTURE 2: HOVER MODE (Index + Middle UP: ✌️)
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                current_gesture = "HOVER / LIFT PEN ✌️"
                gesture_color = (255, 200, 0)
                status_msg = "Pen Lifted (Move cursor freely)"
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (x_idx, y_idx), 10, (255, 200, 0), 2)

            # GESTURE 3: CLEAR (All 5 fingers OPEN ✋)
            elif sum(fingers) >= 4:
                current_gesture = "CLEAR CANVAS ✋"
                gesture_color = (0, 0, 255)
                status_msg = "Canvas Cleared!"
                draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_x, prev_y = 0, 0
                s_digit = None
                probabilities = np.zeros(10)

            # GESTURE 4: PREDICT (Fist ✊ or Thumb UP 👍)
            elif sum(fingers) == 0 or fingers == [1, 0, 0, 0, 0]:
                current_gesture = "PREDICTING 🎯"
                gesture_color = (255, 0, 255)
                prev_x, prev_y = 0, 0

                gray_c = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                if np.count_nonzero(gray_c) > 60:
                    c28, vec, _ = preprocess_digit_roi(draw_canvas)
                    if vec is not None:
                        last_canvas_28 = c28
                        s_probs = softmax_model.predict(vec, verbose=0)[0]
                        s_digit = int(np.argmax(s_probs))
                        s_conf = float(s_probs[s_digit] * 100)
                        probabilities = s_probs

                        if cnn_model:
                            c_in = np.expand_dims(c28.astype("float32")/255.0, (0, -1))
                            c_probs = cnn_model.predict(c_in, verbose=0)[0]
                            c_digit = int(np.argmax(c_probs))
                            c_conf = float(c_probs[c_digit] * 100)

                        status_msg = f"Softmax: {s_digit} ({s_conf:.1f}%) | CNN: {c_digit} ({c_conf:.1f}%)"
                else:
                    status_msg = "Draw a digit first!"

            else:
                prev_x, prev_y = 0, 0

        else:
            prev_x, prev_y = 0, 0

        # Composite drawn strokes with camera feed
        gray_canvas = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
        _, stroke_mask = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(stroke_mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        stroke_fg = cv2.bitwise_and(draw_canvas, draw_canvas, mask=stroke_mask)
        stroke_fg[:, :, 0] = np.where(stroke_fg[:, :, 0] > 0, 255, 0)
        stroke_fg[:, :, 1] = np.where(stroke_fg[:, :, 1] > 0, 255, 0)
        stroke_fg[:, :, 2] = 0

        merged = cv2.add(frame_bg, stroke_fg)

        # Draw HUD overlays and buttons
        draw_hud(merged, s_digit, s_conf, c_digit, c_conf, probabilities, status_msg, current_gesture, gesture_color)

        # Mini 28x28 Preprocessed Input preview in corner
        preview_scaled = cv2.resize(last_canvas_28, (84, 84), interpolation=cv2.INTER_NEAREST)
        preview_bgr = cv2.cvtColor(preview_scaled, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview_bgr, (0, 0), (83, 83), (0, 255, 0), 1)
        merged[h - 94 : h - 10, 10 : 94] = preview_bgr
        cv2.putText(merged, "28x28 Model Input", (10, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Air Canvas - 5-Finger Gesture Digit Recognition", merged)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            s_digit = None
            probabilities = np.zeros(10)
            status_msg = "Canvas Cleared"
        elif key == ord('p') or key == 32:
            gray_c = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
            if np.count_nonzero(gray_c) > 60:
                c28, vec, _ = preprocess_digit_roi(draw_canvas)
                if vec is not None:
                    last_canvas_28 = c28
                    s_probs = softmax_model.predict(vec, verbose=0)[0]
                    s_digit = int(np.argmax(s_probs))
                    s_conf = float(s_probs[s_digit] * 100)
                    probabilities = s_probs
                    if cnn_model:
                        c_in = np.expand_dims(c28.astype("float32")/255.0, (0, -1))
                        c_probs = cnn_model.predict(c_in, verbose=0)[0]
                        c_digit = int(np.argmax(c_probs))
                        c_conf = float(c_probs[c_digit] * 100)
                    status_msg = f"Softmax: {s_digit} ({s_conf:.1f}%) | CNN: {c_digit} ({c_conf:.1f}%)"
        elif key == ord('s'):
            snap_path = os.path.join(PROJECT_ROOT, "outputs", "air_canvas_snapshot.png")
            cv2.imwrite(snap_path, merged)
            print(f"[+] Saved snapshot to: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Air Canvas session ended.")


if __name__ == "__main__":
    run_air_canvas()
