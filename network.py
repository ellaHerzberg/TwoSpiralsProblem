from layer import Layer
import numpy as np
from constants import ETA


class Network(object):
    def __init__(self, input_size, eta, first_activation):
        self.eta = eta
        self.layers = [Layer(input_size, first_activation)]
        self.layers[0].nodes = np.random.uniform(-1, 1, input_size)

    def add_layer(self, n_new_nodes, activ_func):
        last_layer = self.layers[-1]
        random_weights = []
        for i in range(n_new_nodes):
            node_random_weights = np.random.uniform(-1, 1, last_layer.n_nodes)
            random_weights.append(node_random_weights)
        last_layer.weights = random_weights
        self.layers.append(Layer(n_new_nodes, activ_func))

    def compute_error(self, inputs, teacher_answer):
        return self.predict(inputs) - teacher_answer

    def predict(self, inputs):
        self.layers[0].nodes = inputs

        for l in range(len(self.layers) - 1):
            old_layer = self.layers[l]
            new_layer = self.layers[l + 1]

            new_layer_nodes = []
            new_layer_d_nodes = []
            for i in range(new_layer.n_nodes):
                curr_node = 0
                curr_d_node = 0
                for j in range(old_layer.n_nodes):
                    curr_node += old_layer.activ_func.activation(old_layer.weights[i][j] * old_layer.nodes[j])
                    curr_d_node += old_layer.activ_func.derivative(old_layer.weights[i][j] * old_layer.nodes[j])
                new_layer_nodes.append(curr_node)
                new_layer_d_nodes.append(curr_d_node)
            new_layer.nodes = new_layer_nodes
            new_layer.d_nodes = new_layer_d_nodes

        return self.layers[-1].nodes[0]

    def update_weights(self, inputs, teacher_answer):
        error = self.compute_error(inputs, teacher_answer)
        last_delta = error * self.layers[-1].d_nodes[0]
        last_d_weights = np.multiply(self.layers[-2].nodes, (last_delta * -ETA))
        self.layers[-2].weights = np.add(self.layers[-2].weights, last_d_weights)

        for l in range(len(self.layers) - 2, 0, -1):

            curr_layer = self.layers[l]
            next_layer = self.layers[l - 1]

            curr_delta = np.multiply(np.multiply(last_delta, curr_layer.weights), curr_layer.nodes)
            curr_d_weights = np.multiply(np.array([next_layer.nodes]).transpose(), curr_delta) * -ETA

            # # Compute delta
            # next_delta = last_delta * curr_layer.activ_func.derivative(curr_layer)
            # h = self.layers[1].nodes
            # dJ = np.multiply(h, -self.eta * delta2)
            #
            # hp = np.multiply(h, np.subtract(1, h))
            # delta1 = np.multiply(hp, np.multiply(J, delta2))
            # dW = np.multiply(np.array(inputs).reshape(len(inputs), 1), -self.eta * delta1)

            # Update weights
            self.layers[l - 1].weights = np.add(self.layers[0].weights, curr_d_weights.transpose())
