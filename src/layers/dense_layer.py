import numpy as np

class DenseLayer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.inputs = None
        self.outputs = None

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
        self.inputs = input_data
        self.outputs = np.dot(input_data, self.weights) + self.biases
        return self.outputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs