import numpy as np

class ReLU:
 
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs