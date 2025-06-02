# Layer Module

## Overview

This directory implements the **core layer abstractions** of a fully-connected feedforward neural network, developed entirely from scratch using only NumPy. The foundational building block currently included is the `DenseLayer` (also known as a fully connected layer). Future extensions to this module will include layers such as `Dropout` to support regularization techniques.

The purpose of this module is to encapsulate the fundamental operations required for forward and backward propagation within multilayer neural architectures. Each layer is designed for full interoperability with the computational graph used in gradient-based optimization.

---

## 1. Dense Layer

### Description

A **dense layer** is a linear transformation defined by a weight matrix \( W \in \mathbb{R}^{n_{\text{in}} \times n_{\text{out}}} \) and a bias vector \( b \in \mathbb{R}^{1 \times n_{\text{out}}} \). For an input matrix \( X \in \mathbb{R}^{m \times n_{\text{in}}} \) (where \( m \) is the batch size), the forward pass computes the output:

\[
Y = XW + b
\]

This linear operation is central to deep learning architectures, as it provides a learned affine transformation that is typically followed by a nonlinear activation.

### Forward Propagation

The method `forward(input_data)` performs the linear transformation described above. It accepts a 2D NumPy array with shape:

- \( (m, n_{\text{in}}) \): batch of \( m \) input vectors of dimension \( n_{\text{in}} \)

It computes:

\[
\text{outputs} = \text{inputs} \cdot \text{weights} + \text{biases}
\]

This output is then typically passed to a nonlinear activation function (not handled at the layer level).

### Backward Propagation

The method `backward(dvalues)` computes the gradients necessary for the backpropagation algorithm, which is the backbone of stochastic gradient descent.

Given the gradient of the loss function with respect to the layer’s outputs, \( \frac{\partial L}{\partial Y} \), it computes:

- \( \frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} \)
- \( \frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial Y} \) (across the batch)
- \( \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T \)

These expressions are derived directly from the chain rule applied to:

\[
Y = XW + b
\]

These gradients are stored as internal attributes (`dweights`, `dbiases`, `dinputs`) to be used by the optimizer for parameter updates.

---

## 2. Design Philosophy

This implementation intentionally avoids abstraction overhead, making it easier to trace every computation from inputs to gradients. It is suitable for:

- Educational purposes, where clarity and mathematical transparency are paramount
- Research prototypes, where precise control over computational flow is required
- Experimental architectures, where custom layer definitions are needed

All tensors are assumed to be NumPy arrays. No automatic broadcasting or shape inference is done beyond strict shape validation, ensuring consistency and reproducibility.

---

## 3. Future Directions

Upcoming enhancements to this module will include:

- `DropoutLayer`: To support stochastic regularization during training
- `BatchNormalizationLayer`: For improving training dynamics
- Layer wrappers for easy network construction

---

## 4. Dependencies

This module depends solely on:

- `numpy`: for matrix operations and numerical computation

No external machine learning frameworks are used, ensuring full transparency and understanding of underlying operations.


