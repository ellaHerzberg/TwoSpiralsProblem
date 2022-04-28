from random import randint
from layer import Layer
import numpy as np


class Network(object):
    def __init__(self, input_size, learning_rate):
        self.input_layer = [randint(-1, 1)] * input_size
        self.output_layer = -1
        self.hidden_layers = []
        self.learning_rate = learning_rate

    def add_layer(self, num_of_nodes, activation):
        random_weights = [randint(-1, 1)] * num_of_nodes
        layer = Layer(random_weights, activation)
        self.hidden_layers.append(layer)

    def connect_layers(self, layer_1, layer_2):
        pass


    def error_function(self):
        # self.output_layer -
        pass