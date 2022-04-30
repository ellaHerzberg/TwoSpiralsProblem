from layer import Layer
import numpy as np


class Network(object):
    def __init__(self, input_size, eta):
        self.eta = eta
        self.layers = [Layer(2, "sigmoid")]
        self.layers[0].nodes = np.random.uniform(-1, 1, input_size)

    def add_layer(self, n_new_nodes, activation):
        last_layer = self.layers[-1]
        random_weights = []
        for i in range(n_new_nodes):
            node_random_weights = np.random.uniform(-1, 1, last_layer.n_nodes)
            random_weights.append(node_random_weights)
        last_layer.weights = random_weights
        self.layers.append(Layer(n_new_nodes, activation))

    def compute_error(self, inputs, teacher_answer):
        return self.predict(inputs) - teacher_answer

    def predict(self, inputs):
        self.layers[0].nodes = inputs

        for l in range(2):
            old_layer = self.layers[l]
            new_layer = self.layers[l + 1]

            new_layer_nodes = []
            for i in range(new_layer.n_nodes):
                curr_node = 0
                for j in range(old_layer.n_nodes):
                    curr_node += old_layer.activation(old_layer.weights[i][j] * old_layer.nodes[j])
                new_layer_nodes.append(curr_node)
            new_layer.nodes = new_layer_nodes

        return self.layers[-1].nodes[0]

    def update_weights(self, inputs, teacher_answer):
        # Not generic at all. #TODO: Make it generic

        W = self.layers[0].weights
        J = self.layers[1].weights
        error = self.compute_error(inputs, teacher_answer)

        # Compute deltas
        delta2 = error * 1
        h = self.layers[1].nodes
        dJ = np.multiply(h, -self.eta * delta2)

        hp = np.multiply(h, np.subtract(1, h))
        delta1 = np.multiply(hp, np.multiply(J, delta2))
        dW = np.multiply(np.array(inputs).reshape(len(inputs), 1), -self.eta * delta1)

        # Update weights
        self.layers[1].weights = np.add(self.layers[1].weights, dJ)
        self.layers[0].weights = np.add(self.layers[0].weights, dW.transpose())
