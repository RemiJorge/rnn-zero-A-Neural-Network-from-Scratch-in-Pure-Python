import numpy as np


class Accuracy:
    """
    Accuracy metric class.
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def calculate(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        predictions = np.argmax(y_pred, axis=1)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        accuracy = np.mean(predictions == y_true)
        return accuracy