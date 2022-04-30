from constants import EPOCH_SIZE

def check_network(net, prediction_percentage, validation_data, curr_index):

    # Check Network
    correct_0 = 0
    correct_1 = 0
    false_0 = 0
    false_1 = 0

    for r in validation_data:
        x, y, teacher = r.split(',')
        x = float(x)
        y = float(y)
        teacher = int(teacher[0])

        # Calculate accuracy
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
                pass
        else:
            raise ValueError("Teacher is not 0 / 1")

    # Sum accuracy
    corrects = correct_1 + correct_0
    falses = false_1 + false_0

    curr_prediction_percent = (corrects / (falses + corrects))
    prediction_percentage.append(curr_prediction_percent)

    # print(curr_prediction_percent)
    if curr_index % EPOCH_SIZE == 0:
        curr_epoch = prediction_percentage[-EPOCH_SIZE:]
        print("Epoch {epoch_n} had prediction of {epoch_pred:.2f}".format(epoch_n=(curr_index / EPOCH_SIZE), epoch_pred=(sum(curr_epoch) / len(curr_epoch))))
