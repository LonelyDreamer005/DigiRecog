"""
preprocess.py
-------------
Robust Image Preprocessing Pipeline for Handwritten Digits.

Handles both:
1. Physical Camera Captures (dark ink on bright paper with shadows/lighting)
2. Digital / Air Canvas Captures (white strokes on dark canvas)

Transforms any input into MNIST-standard format:
- Isolated digit bounding box
- Preserved aspect ratio scaled into a 20x20 bounding box
- Centered on a 28x28 black canvas (4px padding)
- White digit (255 / 1.0) on pure Black background (0 / 0.0)
- Normalized float32 vector (1, 784) for Softmax and tensor (1, 28, 28, 1) for CNN
"""

import cv2
import numpy as np


def preprocess_digit_roi(roi_bgr, target_size=(28, 28)):
    """
    Preprocesses a Region of Interest (ROI) from webcam, canvas, or air-drawing.
    
    Parameters:
        roi_bgr (numpy.ndarray): BGR or Grayscale image containing the drawn digit.
        target_size (tuple): Desired output dimensions (28, 28).
        
    Returns:
        tuple: (canvas_28x28 uint8, normalized_vector_784 float32, debug_info dict)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros(target_size, dtype=np.uint8), np.zeros((1, 784), dtype=np.float32), {"error": "Empty ROI"}

    # Step 1: Convert to Grayscale
    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr.copy()

    # Step 2: Auto-detect background polarity
    # If the majority of corner/border pixels are dark, it is a dark canvas with bright strokes (Air Canvas / Draw Canvas).
    # If the majority of pixels are bright, it is a white paper capture with dark ink.
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1], gray[0, gray.shape[1]//2]]
    avg_corner = np.mean(corners)

    if avg_corner > 127:
        # Paper capture: Dark ink on white paper -> Invert with Otsu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Air/Digital Canvas: Bright strokes on dark background -> Direct thresholding (NO invert)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Morphological closing to seal small pen stroke breaks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 3: Find Contours of the drawn strokes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas_28x28 = np.zeros(target_size, dtype=np.uint8)

    if contours:
        # Find the overall bounding box enclosing all major stroke contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
        if not valid_contours:
            valid_contours = contours

        # Combine all valid stroke contours to get complete digit bounding box
        all_pts = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_pts)

        if w > 4 and h > 4:
            digit_crop = thresh[y : y + h, x : x + w]

            # Step 4: Aspect-ratio preserving scale into a 20x20 box
            if w > h:
                new_w = 20
                new_h = max(1, int(round((h * 20.0) / w)))
            else:
                new_h = 20
                new_w = max(1, int(round((w * 20.0) / h)))

            resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Step 5: Center on a 28x28 black canvas
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            canvas_28x28[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_digit

    # Step 6: Normalize to [0.0, 1.0]
    normalized = canvas_28x28.astype("float32") / 255.0
    normalized_vector = normalized.reshape((1, 784))

    debug_info = {
        "thresh": thresh,
        "canvas_28x28": canvas_28x28,
        "num_contours": len(contours)
    }

    return canvas_28x28, normalized_vector, debug_info
