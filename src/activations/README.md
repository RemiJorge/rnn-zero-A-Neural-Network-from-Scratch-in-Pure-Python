# Activation Functions Module

## Overview

This directory contains implementations of **activation functions** used in neural networks. Activation functions introduce non-linearity, allowing the network to approximate complex functions and learn meaningful patterns beyond linear transformations. 

Two fundamental activations are currently included:

- **ReLU (Rectified Linear Unit)**: Common in hidden layers for its simplicity and efficiency.
- **Softmax**: Used in output layers for multi-class classification problems.

This module is part of a complete neural network implementation from scratch using only NumPy. Each class encapsulates the forward and backward passes required for integration into gradient-based learning.

---

## 1. ReLU Activation

### Description

The **ReLU (Rectified Linear Unit)** function is defined as:

\[
f(x) = \max(0, x)
\]

This simple yet effective non-linearity is widely used in deep neural networks due to its sparsity-inducing and computationally efficient behavior. It enables the network to converge faster and alleviates the vanishing gradient problem.

### Forward Propagation

Given input tensor \( X \in \mathbb{R}^{m \times n} \), the ReLU activation is applied element-wise:

\[
Y_{ij} = \begin{cases}
X_{ij} & \text{if } X_{ij} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

This operation preserves only positive activations, effectively introducing piecewise linearity into the model.

### Backward Propagation

During the backward pass, the gradient is propagated only through the elements of the input that were strictly positive in the forward pass:

\[
\frac{\partial L}{\partial X_{ij}} = \begin{cases}
\frac{\partial L}{\partial Y_{ij}} & \text{if } X_{ij} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

This is implemented efficiently using NumPy array operations.

---

## 2. Softmax Activation

### Description

The **Softmax** function maps a real-valued vector to a probability distribution over mutually exclusive classes. For an input vector \( z \in \mathbb{R}^K \), the softmax function is defined as:

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

This ensures that all outputs are in the range \( (0,1) \) and sum to 1, making them interpretable as probabilities.

### Forward Propagation

For a batch input \( Z \in \mathbb{R}^{m \times K} \):

- Numerical stability is ensured by subtracting \( \max(Z, \text{axis}=1) \) before exponentiation.
- The output is a matrix of shape \( (m, K) \) where each row is a valid probability distribution.

\[
\text{Softmax}(Z_i) = \frac{\exp(Z_i - \max(Z_i))}{\sum_j \exp(Z_{ij} - \max(Z_i))}
\]

### Backward Propagation

The derivative of the softmax function is a Jacobian matrix:

\[
\frac{\partial \text{Softmax}_i}{\partial z_j} = \text{Softmax}_i \cdot (\delta_{ij} - \text{Softmax}_j)
\]

For general loss functions, computing this gradient explicitly requires handling the Jacobian. However, in practice, the backward method is typically implemented in combination with the cross-entropy loss, where the expression simplifies considerably.

*Note:* The current implementation leaves `backward()` unimplemented due to its complexity when used in isolation.

---

## 3. Design Philosophy

Each activation function is encapsulated in a class with:

- A `forward()` method that performs the transformation.
- A `backward()` method that computes the gradient of the loss with respect to the input.
- Internal state storage (`self.inputs`, `self.outputs`) to support backpropagation.

This modular design enables flexible integration into a layer-wise architecture and clear debugging during training.

---

## 4. Future Extensions

Planned additions to the activation module include:

- **Sigmoid**: Useful for binary classification or gating mechanisms.
- **Tanh**: Zero-centered alternative to sigmoid.
- **Leaky ReLU / Parametric ReLU**: Variants of ReLU that avoid dead neurons.

Backward propagation for `Softmax` will be completed in tandem with the cross-entropy loss implementation.
