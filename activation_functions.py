import numpy as np


class Linear(object):
    def __init__(self):
        self.activation = lambda x : x
        self.derivative = lambda x : 1

class Sigmoid(object):
    def __init__(self):
        self.activation = lambda x : 1 / (1 + np.exp(-x))
        self.derivative = lambda x : x * (1 - x)

class Relu(object):
    def __init__(self):
        self.activation = lambda x : x * (x > 0)
        self.derivative = lambda x : 1. * (x > 0)

def tanh(x):
    return np.tanh(x)


