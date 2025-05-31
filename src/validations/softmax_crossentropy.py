import numpy as np
from src.activations.softmax import Softmax
from src.validations.crossentropy import Crossentropy

class SoftmaxCrossentropy:

    def __init__(self):
        self.activation = Softmax()
        self.loss = Crossentropy()
        self.outputs = None

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.output
        return self.loss.calculate(self.outputs, y_true)

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs