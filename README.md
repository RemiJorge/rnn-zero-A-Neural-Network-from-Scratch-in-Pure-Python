# Neural Network From Scratch 🧠

> A minimalist and educational neural network library built entirely with NumPy.

This repository demonstrates the inner workings of neural networks **without using high-level machine learning libraries** such as TensorFlow or PyTorch. The goal is to provide full transparency and didactic clarity in understanding how each component works under the hood.

---

## 🌲 Repository Structure

.
├── example.ipynb               # Example: simple training on toy dataset
├── neural\_network.py           # NeuralNetwork class: high-level training API
├── requirements.txt            # Required Python packages
├── src/
│   ├── activations/            # Activation functions (ReLU, Softmax)
│   ├── layers/                 # Layer definitions (e.g., Dense)
│   ├── optimizers/             # Optimizers (SGD, Adam, RMSprop, Adagrad)
│   └── validations/            # Accuracy and loss computations
└── README.md                   # Main project README


---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/RemiJorge/rnn-zero-A-Neural-Network-from-Scratch-in-Pure-Python.git
cd rnn-zero-a-neural-network-from-scratch-in-pure-python
pip install -r requirements.txt
````

### Run the Example

Open the notebook:

```bash
jupyter notebook example.ipynb
```

This walks you through a complete training loop with visualization.

---

## 🧩 Core Components

The implementation is split across several cleanly modular folders. Each has its own dedicated README for details:

| Module                                         | Description                                                    | Link |
| ---------------------------------------------- | -------------------------------------------------------------- | ---- |
| **[activations](./src/activations/README.md)** | Implements ReLU and Softmax for forward/backward propagation   | 🔗   |
| **[layers](./src/layers/README.md)**           | Defines fully-connected (dense) layer mechanics                | 🔗   |
| **[optimizers](./src/optimizers/README.md)**   | Includes SGD, Adam, RMSprop, and Adagrad optimizers            | 🔗   |
| **[validations](./src/validations/README.md)** | Accuracy and loss (CrossEntropy, Softmax-CrossEntropy) metrics | 🔗   |

---

## 🧠 NeuralNetwork Class

The high-level API is provided by `neural_network.py`, which:

* Orchestrates the forward and backward passes
* Combines loss and accuracy calculations
* Updates parameters using your choice of optimizer
* Supports validation splits

Usage example:

```python
from src.layers.dense_layer import DenseLayer
from src.activations.relu import ReLU
from src.optimizers.adam import Optimizer_Adam
from neural_network import NeuralNetwork

# Define layers
layers = [
    DenseLayer(2, 64),
    ReLU(),
    DenseLayer(64, 3)
]

# Instantiate model
model = NeuralNetwork(layers=layers, optimizer=Optimizer_Adam())

# Train
loss, acc, val_loss, val_acc = model.train(X, y, validation=0.1)
```

---

## 🧪 Requirements

* Python ≥ 3.8
* NumPy
* Jupyter (optional, for running notebooks)

Install them via:

```bash
pip install -r requirements.txt
```

---

## 📜 License

This work is licensed under a
**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
See [LICENSE](https://creativecommons.org/licenses/by-nc-nd/4.0/) for details.

---

## 🙏 Acknowledgments

Inspired by the incredible didactic work of [Vizuara on YouTube](https://www.youtube.com/@Vizuara).

---

## 📬 Contribute

This project is intended for learning and experimentation. Feel free to fork and extend it with:

* More activation functions
* Batch training
* Regularization
* Custom datasets

---

## 📚 Related Topics

* Backpropagation algorithm
* Gradient descent variants
* Activation function behavior
* Loss metrics and stability
* Data preprocessing for classification

---

Happy learning! 🌱
