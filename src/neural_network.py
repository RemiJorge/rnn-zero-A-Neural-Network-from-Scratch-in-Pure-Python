from src.activations.relu import ReLU
from src.activations.softmax import Softmax

from src.layers.dense_layer import DenseLayer

from src.optimizers.adagrad import Optimizer_Adagrad
from src.optimizers.adam import Optimizer_Adam
from src.optimizers.gradient import Optimizer_SGD
from src.optimizers.rmsprop import Optimizer_RMSprop

from src.validations.softmax_crossentropy import SoftmaxCrossentropy
from src.validations.accuracy import Accuracy


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
        self.loss = None

    def forward(self, inputs):

        self.inputs = inputs
        self.outputs = inputs.copy()

        for layer in self.layers:
            self.outputs = layer.forward(self.outputs)


    def backward(self, dinputs):

        self.dinputs = dinputs.copy()

        for layer in self.layers[::-1]:
            self.dinputs = layer.backward(self.dinputs)


    def train(self, inputs, y_true):

        self.forward(inputs)
        self.loss = self.softmax_loss.forward(self.outputs, y_true)
        self.acc = self.accuracy.calculate(inputs, y_true)
        
        self.dinputs = self.softmax_loss.backward(self.outputs, y_true)
        self.backward(self.dinputs)
        self.update_optimizer()


    def update_optimizer(self):
        self.optimizer.pre_update_params()
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()


    def calculate(self, inputs):

        self.forward(inputs)
        
        return self.softmax.forward(self.outputs)

