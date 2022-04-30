# using python 3.8

import matplotlib.pyplot as plt
import numpy as np
from network import Network
from constants import *

net = Network(INPUT_SIZE, ETA)
net.add_layer(3, LINEAR)
net.add_layer(1, LINEAR)

with open(DATA_TRAIN_PATH) as data_file:
    data = data_file.readlines()

# training the network
for r in data:
    x, y, teacher = r.split(',')
    x = float(x)
    y = float(y)
    teacher = int(teacher[0])

    net.update_weights([x, y], teacher)

# check if it works
with open(DATA_VALID_PATH) as data_file:
    data = data_file.readlines()

wins = 0
loses = 0
x_array_zero = np.array([])
x_array_one = np.array([])
y_array_zero = np.array([])
y_array_one = np.array([])
for r in data:
    x, y, teacher = r.split(',')
    x = float(x)
    y = float(y)
    teacher = int(teacher[0])

    ans = round(net.predict([x, y]))
    # create vectors for visualization
    if ans == 0:
        x_array_zero = np.append(x_array_zero, x)
        y_array_zero = np.append(y_array_zero, y)
    if ans == 1:
        x_array_one = np.append(x_array_one, x)
        y_array_one = np.append(y_array_one, y)

    if ans == teacher:
        wins += 1
    else:
        loses += 1

print(wins / (loses + wins))
plt.scatter(x_array_zero, y_array_zero, color='#e619ae')
plt.scatter(x_array_one, y_array_one, color='#AEE619')
plt.show()
