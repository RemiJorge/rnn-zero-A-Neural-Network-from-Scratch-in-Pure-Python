from src.activations.relu import ReLU
from src.activations.softmax import Softmax

from src.layers.dense_layer import DenseLayer

from src.optimizers.adagrad import Optimizer_Adagrad
from src.optimizers.adam import Optimizer_Adam
from src.optimizers.gradient import Optimizer_SGD
from src.optimizers.rmsprop import Optimizer_RMSprop

from src.validations.softmax_crossentropy import SoftmaxCrossentropy
from src.validations.accuracy import Accuracy

import numpy as np

class NeuralNetwork:

    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

        # for training
        self.softmax_loss = SoftmaxCrossentropy()
        # for calculation
        self.softmax = Softmax()
        
        self.accuracy = Accuracy()

        self.inputs = None
        self.outputs = None

    def forward(self, inputs):

        self.inputs = inputs
        self.outputs = inputs.copy()
        for i in range(len(self.layers)):
            self.outputs = self.layers[i].forward(self.outputs)


    def backward(self, dinputs):

        self.dinputs = dinputs.copy()
        for i in range(len(self.layers)-1, -1, -1):
            self.dinputs = self.layers[i].backward(self.dinputs)


    def update_optimizer(self):
        self.optimizer.pre_update_params()
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()


    def train(self, inputs, y_true, validation=0.1):

        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = inputs[indices]
        y_true = y_true[indices]
        split_index = int(len(inputs) * (1 - validation))
        x_train, x_val = inputs[:split_index], inputs[split_index:]
        y_train, y_val = y_true[:split_index], y_true[split_index:]

        self.forward(x_train)
        loss = self.softmax_loss.forward(self.outputs, y_train)
        acc = self.accuracy.calculate(self.softmax_loss.outputs, y_train)
        
        self.dinputs = self.softmax_loss.backward(self.softmax_loss.outputs, y_train)
        self.backward(self.dinputs)
        self.update_optimizer()

        if validation > 0:
            self.forward(x_val)
            val_loss = self.softmax_loss.forward(self.outputs, y_val)
            val_acc = self.accuracy.calculate(self.softmax_loss.outputs, y_val)
            return loss, acc, val_loss, val_acc

        return loss, acc


    def calculate(self, inputs):

        self.forward(inputs)
        
        return self.softmax.forward(self.outputs)

    def get_current_learning_rate(self):
        return self.optimizer.current_learning_rate
    