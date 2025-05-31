import numpy as np

class ReLU:
 
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, input_data):
        self.inputs = input_data
        self.outputs = np.maximum(0, input_data)
        return self.outputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs