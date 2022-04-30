# using python 3.8
from network import Network
from random import shuffle
from check_network import check_network
from constants import *
from visualize import *

def main():
    prediction_percentage = []

    net = Network(INPUT_SIZE, ETA, SIGMOID)
    net.add_layer(30, LINEAR)
    net.add_layer(1, LINEAR)

    # Open Training data
    with open(DATA_TRAIN_PATH) as train_data_file, open(DATA_VALID_PATH) as validation_data_file:
        train_data = train_data_file.readlines()
        validation_data = validation_data_file.readlines()
    # Loop over each example
    for i in range(LEARNING_LOOPS):
        shuffle(train_data)

        # Train the network
        train_network(train_data, net)
        check_network(net, prediction_percentage, validation_data, i)

    last_epoch = sum(prediction_percentage[-EPOCH_SIZE:]) / len(prediction_percentage[-EPOCH_SIZE:])
    visualize(validation_data, net, last_epoch)


def train_network(train_data, net):
    for r in train_data:
        x, y, teacher = r.split(',')
        x = float(x)
        y = float(y)
        teacher = int(teacher[0])
        net.update_weights([x, y], teacher)


if __name__ == "__main__":
    main()