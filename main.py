# using python 3.8

from network import Network

net = Network(2, 0.001)
net.add_layer(3, "linear")
net.add_layer(1, "linear")

with open("./src/DATA_TRAIN.csv") as data_file:
    data = data_file.readlines()

for r in data:
    x, y, teacher = r.split(',')
    x = float(x)
    y = float(y)
    teacher = int(teacher[0])

    net.update_weights([x, y], teacher)


with open("./src/DATA_valid.csv") as data_file:
    data = data_file.readlines()

wins = 0
loses = 0
for r in data:
    x, y, teacher = r.split(',')
    x = float(x)
    y = float(y)
    teacher = int(teacher[0])

    if round(net.predict([x, y])) == teacher:
        wins += 1
    else:
        loses += 1

print(wins / (loses + wins))