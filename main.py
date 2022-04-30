# using python 3.8

import numpy as np
import pickle
import matplotlib
from network import Network


net = Network(2, 0.01)
net.add_layer(3, "linear")
net.add_layer(1, "linear")

net.update_weights([0, 1], 0)
