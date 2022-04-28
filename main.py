# using python 3.8

import numpy as np
import pickle
import matplotlib
from network import Network


net = Network(2, 0.1)
net.add_layer(3, "linear")
print(net.compute_error([0, 1], 0))
net.update_weights([0, 1], 0)
