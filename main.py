# using python 3.8

import matplotlib.pyplot as plt
import numpy as np
from network import Network
from random import shuffle
from check_network import check_network
from constants import *

prediction_percentage = []

net = Network(INPUT_SIZE, ETA, LINEAR)
net.add_layer(30, SIGMOID)
net.add_layer(1, LINEAR)

# Open Training data
with open(DATA_TRAIN_PATH) as data_file:
    train_data = data_file.readlines()
with open(DATA_VALID_PATH) as data_file:
    validation_data = data_file.readlines()
# Loop over each example
for i in range(LEARNING_LOOPS):
    shuffle(train_data)

    # Train the network
    for r in train_data:
        x, y, teacher = r.split(',')
        x = float(x)
        y = float(y)
        teacher = int(teacher[0])
        net.update_weights([x, y], teacher)
        
    check_network(net, prediction_percentage, validation_data, i)


# visualize last run
x_array_zero = np.array([])
x_array_one = np.array([])
y_array_zero = np.array([])
y_array_one = np.array([])
for r in validation_data:
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

plt.scatter(x_array_zero, y_array_zero, color='#e619ae')
plt.scatter(x_array_one, y_array_one, color='#AEE619')
plt.show()
