import activation_functions


class Layer(object):
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = self.set_active_func(activation)
        self.output = 0
#1112
    def set_active_func(self, activation):
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
