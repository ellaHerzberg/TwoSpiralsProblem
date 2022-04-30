import activation_functions
import numpy as np


class Layer(object):
    def __init__(self, n_nodes, activation):
        self.n_nodes = n_nodes
        self.nodes = []
        self.weights = []
        self.activation = self._set_active_func(activation)
        self.output = 0

    def _set_active_func(self, activation):
        if activation == "relu":
            return activation_functions.relu
        elif activation == "tanh":
            return activation_functions.tanh
        elif activation == "drelu":
            return activation_functions.drelu
        elif activation == "linear":
            return activation_functions.linear
        else:
            return activation_functions.sigmoid
