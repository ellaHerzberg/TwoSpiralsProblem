import activation_functions
from constants import *


class Layer(object):
    def __init__(self, n_nodes, activation):
        self.n_nodes = n_nodes
        self.nodes = []
        self.weights = []
        self.activation = self._set_active_func(activation)
        self.output = 0

    def _set_active_func(self, activation):
        if activation == RELU:
            return activation_functions.relu
        elif activation == TANH:
            return activation_functions.tanh
        elif activation == DRULU:
            return activation_functions.drelu
        elif activation == LINEAR:
            return activation_functions.linear
        else:
            return activation_functions.sigmoid
