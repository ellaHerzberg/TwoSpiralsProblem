import numpy as np

from activation_functions import Relu, Sigmoid, Sin, Tanh, Linear
from constants import *


class Layer(object):
    def __init__(self, n_nodes, activ_func):
        self.n_nodes = n_nodes
        self.nodes = np.array([])
        self.d_nodes = np.array([])
        self.weights = np.array([])
        self.activ_func = self._set_active_func(activ_func)
        self.output = 0

    def _set_active_func(self, activ_func):
        if activ_func == SIN:
            return Sin()
        elif activ_func == SIGMOID:
            return Sigmoid()
        elif activ_func == RELU:
            return Relu()
        elif activ_func == TANH:
            return Tanh()
        elif activ_func == LINEAR:
            return Linear()