# Validation & Loss Module

**[Back](../../README.md)**

## Overview

This directory contains core components for **model evaluation and loss computation**, essential for training neural networks using gradient-based optimization. Specifically, it includes:

- **Accuracy**: A performance metric for classification tasks.
- **Cross-Entropy**: A loss function quantifying the dissimilarity between predicted and true distributions.
- **Softmax + Cross-Entropy Fusion**: A numerically stable and efficient composite layer commonly used in multi-class classification settings.

These components are implemented from first principles using NumPy, and are designed to support forward and backward passes for end-to-end differentiable learning.

---

## 1. Accuracy Metric

### Description

The `Accuracy` class computes the **classification accuracy**, defined as the proportion of correct predictions over the total number of samples:

\[
\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}( \hat{y}_i = y_i )
\]

Where \( \hat{y}_i \) is the predicted class label (argmax of predicted probabilities), and \( y_i \) is the true class label.

### Usage

- Supports both integer labels and one-hot encoded labels.
- Converts logits to class predictions via `argmax`.

This metric is ideal for **monitoring performance** during training, validation, and testing phases.

---

## 2. Cross-Entropy Loss

### Description

The **Cross-Entropy Loss** is widely used in classification tasks to measure the distance between the predicted probability distribution \( \hat{y} \) and the true label distribution \( y \). For a single sample, it is given by:

\[
\mathcal{L}(y, \hat{y}) = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
\]

Where \( C \) is the number of classes. When using integer class labels:

\[
\mathcal{L}(y, \hat{y}) = -\log(\hat{y}_{y})
\]

### Forward Propagation

- Applies clipping to \( \hat{y} \) to ensure numerical stability: \( \hat{y} \in [10^{-7}, 1 - 10^{-7}] \)
- Computes negative log-likelihoods per sample
- Returns mean loss across the batch

### Backward Propagation

- Derivative of the loss with respect to predictions \( \hat{y} \):

\[
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
\]

- Normalized by the number of samples for batch averaging

This loss assumes a probabilistic interpretation of the model output (e.g., from a Softmax layer).

---

## 3. Softmax + Cross-Entropy Composite

### Description

The `SoftmaxCrossentropy` class fuses the softmax activation function and the cross-entropy loss into a **single, numerically stable unit**. This fusion avoids the explicit computation of the softmax Jacobian and yields a simpler gradient during backpropagation.

Given logits \( z \in \mathbb{R}^C \), the forward pass computes:

\[
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \quad , \quad \mathcal{L}(y, \hat{y}) = -\log(\hat{y}_y)
\]

### Forward Propagation

- Applies the softmax transformation
- Computes the cross-entropy loss based on the resulting probabilities

### Backward Propagation

The derivative simplifies elegantly:

\[
\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i
\]

This efficient simplification avoids numerical instabilities and reduces computational complexity.

### Practical Notes

- Supports both integer labels and one-hot encodings
- Internally uses `argmax` to resolve class indices when needed

This class is **recommended for final classification layers** in neural networks due to its stability and integration.

---

## 4. Design Philosophy

This module emphasizes:

- **Numerical stability** (clipping, fused operations)
- **Batch-level operations** for efficient vectorized computation
- **Support for both label formats** (sparse and one-hot)

All computations are differentiable and suitable for integration into training loops driven by backpropagation and optimization algorithms.
