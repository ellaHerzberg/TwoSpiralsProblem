import activation_functions


class Layer(object):
    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = None
        self.set_active_func(activation)

    def set_active_func(self, activation):
        if activation == "relu":
            self.activation = activation_functions.relu
        elif activation == "tanh":
            self.activation = activation_functions.tanh
        elif activation == "drelu":
            self.activation = activation_functions.drelu()
        else:
            self.activation = activation_functions.sigmoid
