# using python 3.8

from network import Network
from random import shuffle

LINEAR = "linear"
loop_over_examples = 1000
epoch_size = 10

net = Network(2, 0.001)
net.add_layer(40, LINEAR)
net.add_layer(1, LINEAR)

with open("./src/DATA_TRAIN.csv") as data_file:
    data = data_file.readlines()

prediction_percentage = []
for i in range(loop_over_examples):
    for r in data:
        x, y, teacher = r.split(',')
        x = float(x)
        y = float(y)
        teacher = int(teacher[0])

        net.update_weights([x, y], teacher)

    with open("./src/DATA_valid.csv") as data_file:
        data = data_file.readlines()
        shuffle(data)

    correct_0 = 0
    correct_1 = 0
    false_0 = 0
    false_1 = 0

    for r in data:
        x, y, teacher = r.split(',')
        x = float(x)
        y = float(y)
        teacher = int(teacher[0])

        curr_pred = round(net.predict([x, y]))
        if teacher == 1 == curr_pred:
            correct_1 += 1
        elif teacher == 0 == curr_pred:
            correct_0 += 1
        elif curr_pred != teacher:
            if teacher == 0:
                false_0 += 1
            elif teacher == 1:
                false_1 += 1
            if curr_pred not in [0, 1]:
                # print("Prediction was {pred}".format(pred=curr_pred))
                pass
        else:
            raise ValueError("Teacher is not 0 / 1")

    wins = correct_1 + correct_0
    loses = false_1 + false_0

    curr_prediction_percent = (wins / (loses + wins))
    prediction_percentage.append(curr_prediction_percent)

    # print(curr_prediction_percent)
    if i % epoch_size == 0:
        curr_epoch = prediction_percentage[-epoch_size:]
        print("Epoch {epoch_n} had prediction of {epoch_pred:.2f}".format(epoch_n=(i / epoch_size), epoch_pred=(sum(curr_epoch) / len(curr_epoch))))