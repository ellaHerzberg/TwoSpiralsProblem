from activation_functions import Linear, Sigmoid, Relu
from constants import *


class Layer(object):
    def __init__(self, n_nodes, activ_func):
        self.n_nodes = n_nodes
        self.nodes = []
        self.d_nodes = []
        self.weights = []
        self.activ_func = self._set_active_func(activ_func)
        self.output = 0

    def _set_active_func(self, activ_func):
        if activ_func == LINEAR:
            return Linear()
        elif activ_func == SIGMOID:
            return Sigmoid()
        elif activ_func == RELU:
            return Relu()
