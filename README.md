# Smart Handwritten Digit Recognition with Confidence Estimation and Robustness Analysis

An end-to-end Machine Learning and Deep Learning system for real-time handwritten digit recognition (0–9) on the MNIST dataset. Features a **Softmax Regression** linear baseline compared against a **Convolutional Neural Network (CNN)** extension with live laptop webcam recognition, OpenCV preprocessing, robustness distortion testing, and an interactive Flask dashboard.

---

## 📌 Project Overview & Results Summary

| Metric / Dimension | Softmax Regression (Baseline) | CNN (Deep Learning Extension) | Performance Delta (&Delta;) |
| :--- | :---: | :---: | :---: |
| **Model Type** | Single-layer Linear Classifier | 2 Conv2D + 2 MaxPool + Dense | Deep Spatial Hierarchy |
| **Trainable Parameters** | **7,850** | **225,034** | $+217,184$ |
| **Clean Test Accuracy (10k MNIST)** | **91.88%** | **99.01%** | **$+7.13\%$** (error reduced by $88\%$) |
| **Inference Latency** | **0.63 ms / image** | **0.71 ms / image** | Minimal latency overhead |
| **Gaussian Noise Robustness ($\sigma=0.25$)** | **85.55%** | **98.20%** | $+12.65\%$ |
| **Shift / Translation ($\pm 2$ px)** | **34.25%** | **84.70%** | **$+50.45\%$** (Spatial Invariance) |
| **Rotation ($\pm 15^\circ$)** | **82.85%** | **97.30%** | $+14.45\%$ |
| **Gaussian Blur ($3 \times 3$)** | **91.60%** | **99.00%** | $+7.40\%$ |

---

## 🗂️ Project Directory Structure

```text
DigitRecog/
├── requirements.txt                   # Dependency definitions
├── .gitignore                         # Git exclusion rules
├── README.md                          # Complete project documentation & viva guide
├── models/                            # Saved trained Keras models (.keras)
│   ├── digit_softmax.keras            # Softmax Regression model (7,850 params)
│   └── digit_cnn.keras                # CNN Deep Learning model (225k params)
├── outputs/                           # Experiment outputs and benchmark plots
│   ├── training_curves.png            # Softmax loss and accuracy curves
│   ├── robustness_comparison.png      # Robustness under 5 distortion conditions
│   ├── model_comparison.png           # Softmax vs CNN grouped robustness chart
│   ├── confusion_matrices.png         # Side-by-side Confusion Matrices
│   ├── robustness_results.json        # Raw numerical robustness measurements
│   └── comparison_summary.json        # Model comparison metrics JSON
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── train_softmax.py               # Softmax baseline training on MNIST
│   ├── train_cnn.py                   # CNN deep learning model training
│   ├── robustness.py                  # 5-condition robustness distortion suite
│   ├── compare_models.py              # Full comparison (accuracy, speed, confusion matrix)
│   ├── preprocess.py                  # OpenCV binarization, aspect-ratio scaling & centering
│   ├── webcam_demo.py                 # Standalone OpenCV live webcam application
│   └── app.py                         # Flask web application server
├── templates/
│   └── index.html                     # Responsive glassmorphism web dashboard
└── static/
    ├── css/
    │   └── style.css                  # UI styling with dark mode & glow effects
    └── js/
        └── main.js                    # HTML5 canvas, webcam stream & prediction handling
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup (PowerShell on Windows)

```powershell
# Open terminal in project directory
cd D:\DigitRecog

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Train Models & Run Experiments

```powershell
# 1. Train Softmax Regression Baseline
python src/train_softmax.py

# 2. Train CNN Deep Learning Extension
python src/train_cnn.py

# 3. Run Robustness Distortions Benchmark
python src/robustness.py

# 4. Generate Side-by-Side Model Comparisons & Confusion Matrices
python src/compare_models.py
```

---

### 3. Launch Standalone OpenCV Webcam & Air Canvas Recognition

#### Option A: 🖐️ Air Canvas (5-Finger Gesture In-Air Drawing)
Draw digits directly in the air using hand gestures in front of your laptop camera:
```powershell
python src/air_canvas_demo.py
```
**5-Finger Gestures:**
- ☝️ **Index finger ONLY UP:** `[DRAW MODE]` — Draw strokes in the air with glowing brush.
- ✌️ **Index + Middle UP (Peace Sign):** `[HOVER MODE]` — Move cursor without drawing (lift pen).
- ✋ **All 5 fingers OPEN:** `[CLEAR CANVAS]` — Instantly wipe and erase the canvas.
- ✊ **Fist (0 fingers UP):** `[PREDICT DIGIT]` — Recognize the drawn digit with Softmax & CNN.

#### Option B: Paper-Based Webcam Capture
```powershell
python src/webcam_demo.py
```
**Controls:**
- Position a handwritten digit on white paper inside the green targeting square.
- Press **`s`** to save a snapshot to `outputs/webcam_snapshot.png`.
- Press **`q`** or **`ESC`** to exit.

---

### 4. Launch the Web Interface (Flask)

```powershell
python src/app.py
```
Open your browser at **`http://127.0.0.1:5000`**.

Features available in the web dashboard:
1. **Live Recognition Studio**:
   - Draw on the interactive HTML5 canvas.
   - Capture snapshots directly from your laptop webcam.
   - Upload any custom digit image file.
   - Switch between **Softmax Baseline**, **CNN Model**, or **Compare Both Side-by-Side**.
   - Inspect the live **28×28 Preprocessed Canvas** passed to the model.
2. **Model Comparison Tab**: Inspect side-by-side Confusion Matrices and latency benchmarks.
3. **Robustness Analysis Tab**: View the measured performance table under 5 simulated distortions.
4. **Viva & Theory Guide Tab**: Key questions and mathematical derivations for presentation defense.

---

## 🧠 Key Technical Takeaways for Project Defense

1. **Why is Softmax Regression a Linear Model?**
   Softmax computes $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$, forming hyperplanes in 784-dimensional space. It contains no hidden layers or non-linear feature extractions.

2. **Why does Softmax Fail on Translated Digits ($34.25\%$)?**
   Softmax assigns a static weight $W_{i, j}$ to each specific coordinate $(i, j)$. If a stroke moves by 2 pixels, it activates weights assigned to background or other classes. In contrast, CNN uses localized receptive fields (convolution kernels) and Max-Pooling to achieve spatial translation invariance.

3. **Why Preprocessing with Otsu Thresholding and Centering is Essential:**
   Webcam images differ drastically from MNIST (dark ink on white paper, shadows, variable sizing). The `src/preprocess.py` pipeline converts the image to grayscale, applies Otsu's inverted thresholding, isolates the largest contour, scales the digit preserving its aspect ratio into a $20 \times 20$ box, and centers it on a $28 \times 28$ black canvas.
