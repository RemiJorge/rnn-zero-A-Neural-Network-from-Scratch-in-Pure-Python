# Optimization Algorithms

## Overview

This module implements several foundational **gradient-based optimization algorithms** designed to update the trainable parameters of neural networks efficiently during backpropagation. All optimizers are implemented **from scratch using NumPy**, enabling complete transparency and control over the optimization dynamics.

The algorithms included are:

- **Stochastic Gradient Descent (SGD)** — with optional momentum
- **Adagrad** — with adaptive learning rate
- **RMSprop** — exponential moving average of squared gradients
- **Adam** — adaptive moment estimation combining RMSprop and momentum

Each class supports learning rate decay and state caching for iterative updates across epochs.

---

## 1. Stochastic Gradient Descent (SGD)

### Description

The basic optimizer used in neural networks, **Stochastic Gradient Descent** performs parameter updates using the rule:

\[
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t)
\]

Where \( \eta \) is the learning rate and \( \nabla_{\theta} \mathcal{L} \) is the gradient of the loss function with respect to parameters.

### Momentum Variant

With momentum \( \mu \), the update becomes:

\[
v_t = \mu v_{t-1} - \eta \nabla_{\theta} \mathcal{L}(\theta_t) \\
\theta_{t+1} = \theta_t + v_t
\]

This helps accelerate convergence and smooth noisy gradients.

---

## 2. Adagrad

### Description

**Adagrad** adapts the learning rate individually for each parameter by accumulating the squared gradients:

\[
G_t = G_{t-1} + (\nabla_{\theta} \mathcal{L}(\theta_t))^2 \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla_{\theta} \mathcal{L}(\theta_t)
\]

- \( \epsilon \) is a small constant for numerical stability
- Good for sparse features
- Drawback: aggressive decay in learning rate

### Behavior

Adagrad is effective in scenarios with **sparse gradients**, but may suffer from overly diminishing updates in deep training loops.

---

## 3. RMSprop

### Description

**RMSprop** (Root Mean Square Propagation) resolves Adagrad’s rapid learning rate decay by using an **exponentially decaying average** of squared gradients:

\[
E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho)(\nabla_{\theta} \mathcal{L})^2 \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla_{\theta} \mathcal{L}
\]

- \( \rho \in [0, 1) \) is the decay rate (default: 0.9)
- Maintains a stable update scale
- Works well for non-stationary problems (e.g., RNNs)

---

## 4. Adam

### Description

**Adam (Adaptive Moment Estimation)** combines **momentum** and **RMSprop** by maintaining both the exponentially decaying average of gradients and squared gradients:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{\theta} \mathcal{L} \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_{\theta} \mathcal{L})^2
\]

Bias-corrected estimates:

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

Parameter update rule:

\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\]

- \( \beta_1 = 0.9 \), \( \beta_2 = 0.999 \), \( \epsilon = 10^{-7} \) by default
- State-of-the-art for most deep learning applications

---

## 5. Common Features

All optimizers include:

- **Learning rate decay**: Updated as \( \eta_t = \eta_0 / (1 + \text{decay} \cdot t) \)
- **Numerical stability**: Via additive \( \epsilon \) in denominators
- **Parameter-specific caches**: Momentum and RMS states per layer

