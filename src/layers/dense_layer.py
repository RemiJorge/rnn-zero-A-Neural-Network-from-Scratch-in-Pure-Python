import numpy as np

class DenseLayer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Forward pass through the dense layer.
        :param input_data: Input data of shape (batch_size, input_size)
        :return: Output data of shape (batch_size, output_size)
        """
        if len(input_data.shape) != 2:
            raise ValueError(f"Input data must be 2D, but got {len(input_data.shape)}D")
        if input_data.shape[1] != self.input_size:
            raise ValueError(f"Input data must have shape (batch_size, {self.input_size}), but got {input_data.shape}")
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient):
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update weights and biases using gradients (this is a simple example; in practice, you would use an optimizer)
        learning_rate = 0.01
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient