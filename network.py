from layer import Layer
import numpy as np


class Network(object):
    def __init__(self, input_size, learning_rate):
        self.input_layer = Layer(np.random.uniform(-1, 1, input_size), "sigmoid")
        self.hidden_layers = []

        self.learning_rate = learning_rate

    def add_layer(self, num_of_nodes, activation):
        random_weights = np.random.uniform(-1, 1, num_of_nodes)
        layer = Layer(random_weights, activation)
        self.hidden_layers.append(layer)

    # should be in Layer?
    def compute_error(self, inputs, teacher_answer):
        return self.predict(inputs) - teacher_answer

    def predict(self, inputs):
        first_output = []
        for w in self.input_layer.weights:
            node_output = 0
            for input_node in inputs:
                node_output += self.input_layer.activation(input_node * w)
            first_output.append(node_output)
        self.input_layer.output = first_output

        output = 0
        for i in range(len(first_output)):
            output += first_output[i] * self.hidden_layers[0].weights[i]

        return output

    def update_weights(self, inputs, teacher_answer):

        # Compute deltas
        delta2 = self.compute_error(inputs, teacher_answer) * 1
        J = self.hidden_layers[0].weights
        # Specifically for Sigmoid
        h = self.input_layer.output
        dJ = np.multiply(h, -self.learning_rate * delta2)
        hp = np.multiply(self.input_layer.output, np.subtract(1, h))
        delta1 = np.multiply(hp, delta2 * J)
        dW = np.multiply(inputs, -self.learning_rate * delta1)

        # Update weights
        self.hidden_layers[0].weights = np.add(self.hidden_layers[0].weights, dJ)
        self.input_layer.weights = np.add(self.input_layer.weights, dW)