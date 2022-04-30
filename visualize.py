import numpy as np
import matplotlib.pyplot as plt
from constants import *


def visualize(data):
    """
    plot the received data
    :param data: data to plot
    :return: none
    """
    # set arrays to fill
    x_array_zero = np.array([])
    x_array_one = np.array([])
    y_array_zero = np.array([])
    y_array_one = np.array([])
    # go over the data
    for row in data:
        x, y, teacher = row.split(',')
        x = float(x)
        y = float(y)
        # teacher = int(teacher[0])

        ans = round(net.predict([x, y]))
        # create vectors for visualization
        if ans == 0:
            x_array_zero = np.append(x_array_zero, x)
            y_array_zero = np.append(y_array_zero, y)
        if ans == 1:
            x_array_one = np.append(x_array_one, x)
            y_array_one = np.append(y_array_one, y)

    plt.scatter(x_array_zero, y_array_zero, color=PINK)
    plt.scatter(x_array_one, y_array_one, color=GREEN)
    plt.show()
