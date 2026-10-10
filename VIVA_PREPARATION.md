# Viva Voce & Technical Defense Guide
## Deep Learning CBP: Handwritten Digit Recognition Project

This guide covers typical and challenging questions an examiner or professor may ask during your project presentation/viva.

---

### Part 1: Softmax Regression & Machine Learning Fundamentals

#### Q1: Why is Softmax Regression called "Regression" when it is used for classification?
> **Answer:** It originates from logistic regression and generalized linear models (GLMs). While the final output is a discrete class label, the model internally performs linear regression to compute real-valued continuous scores (logits $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$), which are then mapped to conditional probabilities via the Softmax activation function.

#### Q2: What are the mathematical properties of the Softmax function?
> **Answer:**
> 1. **Non-negativity & Normalization:** $\sum_{k=0}^{K-1} P(y=k \mid \mathbf{x}) = 1$ and $0 \le P(y=k \mid \mathbf{x}) \le 1$.
> 2. **Translation Invariance of Inputs:** Adding a constant $c$ to all logits does not change the resulting probabilities:
>    $$\frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c \cdot e^{z_k}}{e^c \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$
>    *(This property is used in code implementations to subtract $\max(z)$ for numerical stability and prevent floating-point overflow).*

#### Q3: Why did we choose Sparse Categorical Cross-Entropy over Standard Categorical Cross-Entropy or Mean Squared Error (MSE)?
> **Answer:**
> - **vs. MSE:** MSE assumes normally distributed errors and leads to non-convex loss surfaces when paired with sigmoid/softmax, causing slow convergence due to gradient vanishing when predictions are very confident yet wrong. Cross-entropy provides steep gradients proportional to the prediction error $(p_k - y_k)$.
> - **Sparse vs. Standard Cross-Entropy:** Standard categorical cross-entropy requires one-hot encoded label matrices of shape $(N, 10)$, consuming $10\times$ more memory. Sparse cross-entropy accepts integer targets $\{0, \dots, 9\}$ directly and performs index-based lookups, making it faster and memory efficient.

---

### Part 2: Architecture & Deep Learning Comparison (CNN)

#### Q4: Why does Softmax Regression fail on translated/shifted digits (accuracy dropping to 34.25%)?
> **Answer:** Softmax Regression has no concept of 2D spatial locality. It flattens the $28 \times 28$ image into a 784-dimensional 1D vector and learns a single scalar weight for each static pixel coordinate $(i, j)$. If a drawn digit is translated by 2 pixels, the stroke falls on coordinate weights intended for the background or other digits. In contrast, CNN uses **weight sharing across convolutional receptive fields** and **Max Pooling**, providing spatial translation invariance.

#### Q5: Explain the role of Dropout in the CNN extension.
> **Answer:** Dropout randomly deactivates a fraction of neurons (we used $p=0.25$) during each training forward-backward pass. This prevents co-adaptation of feature detectors, forcing individual filters to learn robust, generalized visual cues (such as loop curves and stroke junctions) rather than memorizing training sample noise.

#### Q6: How do the parameter counts compare?
> **Answer:**
> - **Softmax Baseline:** $784 \times 10 + 10 = 7,850$ parameters.
> - **CNN Extension:** $320 \text{ (Conv1)} + 18,496 \text{ (Conv2)} + 204,928 \text{ (Dense1)} + 1,290 \text{ (Dense2)} = \mathbf{225,034}$ parameters.

---

### Part 3: Image Processing & OpenCV Pipeline

#### Q7: Why can't we feed raw webcam images directly to the MNIST models?
> **Answer:** Raw camera frames have major domain shifts compared to MNIST:
> 1. **Polarity:** Pen on paper is dark ink on white background ($0$ on $255$), while MNIST is bright strokes on a black background ($255$ on $0$).
> 2. **Scale & Aspect Ratio:** Handwritten paper digits vary in physical size, whereas MNIST digits are normalized to fit within a $20 \times 20$ bounding box.
> 3. **Positioning:** Webcam digits are unaligned, while MNIST digits are centered inside a $28 \times 28$ canvas.
> 4. **Lighting & Shadows:** Ambient room lighting creates non-uniform illumination.

#### Q8: Walk through the 7 steps in our `preprocess.py` pipeline.
> **Answer:**
> 1. `cv2.cvtColor`: Converts BGR color frame to single-channel Grayscale.
> 2. `cv2.GaussianBlur`: Filters high-frequency paper grain and sensor noise.
> 3. `cv2.threshold` with `THRESH_BINARY_INV + THRESH_OTSU`: Dynamically computes optimal threshold to binarize paper and inverts colors (white digit on black background).
> 4. `cv2.findContours`: Identifies the largest connected ink contour, ignoring stray pen dots.
> 5. Aspect-ratio Preserving Resizing: Computes bounding box $(w, h)$ and scales the larger dimension to 20 pixels while preserving proportions.
> 6. Centering Canvas ($28 \times 28$): Pads the $20 \times 20$ digit with 4-pixel margins on all sides.
> 7. Normalization: Divides $[0, 255]$ by $255.0$ to produce float $[0.0, 1.0]$ tensor.

---

### Part 4: Robustness & Confidence Interpretation

#### Q9: Is Softmax probability equivalent to true model confidence?
> **Answer:** Not necessarily. Softmax is susceptible to **overconfident misclassifications** on out-of-distribution (OOD) inputs because the exponential function forces values to sum to 1 regardless of whether the input resembles any known digit. A truly calibrated uncertainty estimation would require techniques like **Monte Carlo Dropout** or **Deep Ensembles**.

#### Q10: How was the robustness test designed to be scientifically fair?
> **Answer:** We randomly sampled 2,000 test images using a **fixed pseudo-random seed (42)**. The exact same 2,000 images were evaluated across all 5 conditions (Clean, Noise, Shift, Rotation, Blur) for both models. This guarantees that variations in accuracy reflect model robustness to distortions rather than sample selection bias.
