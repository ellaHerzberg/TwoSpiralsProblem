import math

import numpy as np


# class Linear(object):
#     def __init__(self):
#         self.activation = lambda x : x
#         self.derivative = lambda x : 1
#
# class Sigmoid(object):
#     def __init__(self):
#         self.activation = lambda x : 1 / (1 + np.exp(-x))
#         self.derivative = lambda x : self.activation(x) * (1 - self.activation(x))
#
# class Relu(object):
#     def __init__(self):
#         self.activation = lambda x : x * (x > 0)
#         self.derivative = lambda x : 1. * (x > 0)
#
# def tanh(x):
#     return np.tanh(x)


class Relu(object):
    @staticmethod
    def activation(Z):
        """
        This function is the ReLU function.
        :param Z: np.array, numbers to calculate thr ReLU result on.
        :return: np.array, ReLU result.
        """
        return np.maximum(Z, 0)

    @staticmethod
    def derivative(Z):
        """
        This function is the ReLU derivative function.
        :param Z: np.array, numbers to calculate thr ReLU derivative result on.
        :return: np.array, ReLU derivative result.
        """
        return Z > 0


class Sin(object):
    @staticmethod
    def activation(Z):
        """
        This function is the Sin function.
        :param Z: np.array, numbers to calculatete thr Sin result on.
        :return: np.array, Sin result.
        """
        # return np.ravel([np.sin(val) for val in np.ravel(Z)])[np.newaxis].T
        return math.sin(Z)

    @staticmethod
    def derivative(Z):
        """
        This function is the Sin derivative function.
        :param Z: np.array, numbers to calculate thr Sin derivative result on.
        :return: np.array, Sin derivative result.
        """
        # return np.ravel([np.cos(val) for val in np.ravel(Z)])[np.newaxis].T
        return math.cos(Z)


class Sigmoid(object):
    @staticmethod
    def activation(Z):
        """
        This function is the Sigmoid function.
        :param Z: np.array, numbers to calculate thr Sigmoid result on.
        :return: np.array, Sigmoid result.
        """
        return 1 / (1 + np.exp(-Z))

    def derivative(self, Z):
        """
        This function is the Sigmoid derivative function.
        :param Z: np.array, numbers to calculate thr Sigmoid derivative result on.
        :return: np.array, Sigmoid derivative result.
        """
        return self.activation(Z) * (1 - self.activation(Z))

class Tanh(object):
    @staticmethod
    def activation(Z):
        """
        This function is the Tanh function.
        :param Z: np.array, numbers to calculate thr Tanh result on.
        :return: np.array, Tanh result.
        """
        # e = np.exp(Z)
        # e_m = np.exp(-Z)
        # return (e-e_m)/(e+e_m)  # (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return np.tanh(Z)

    def derivative(self, Z):
        """
        This function is the Tanh derivative function.
        :param Z: np.array, numbers to calculate thr Tanh derivative result on.
        :return: np.array, Tanh derivative result.
        """
        return 1-self.activation(Z) ** 2


class Linear(object):
    @staticmethod
    def activation(Z):
        """
        This function is the Linear function.
        :param Z: np.array, numbers to calculate thr Linear result on.
        :return: np.array, Linear result.
        """
        return Z

    def derivative(self, Z):
        """
        This function is the Linear derivative function.
        :param Z: np.array, numbers to calculate thr Linear derivative result on.
        :return: np.array, Linear derivative result.
        """
        return 1
