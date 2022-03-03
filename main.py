import pandas as pd
import numpy as np
import math
from operator import attrgetter


class Weight:
    def __init__(self):
        self.val = 0
        self.idx = 0


def dist(a, b):
    distance = []
    norm = max(max(a), b)
    for elem in a:
        #if type(elem) == type(b):
            if type(elem) == str:
                #print("str dist = ", 0 if (elem == b) else 1)
                distance.append(0 if (elem == b) else 1)
            else:
                #print("this dist = ", math.sqrt(((elem - b) / norm) ** 2))
                distance.append( math.sqrt(((elem - b) / norm) ** 2) )
        #else:
            #print("types neq: type(elem)=",type(elem)," type(b)=", type(b) )
            #print("\n b = ",b)
            #distance.append(1)
    return distance

def predicting (k):
    # =================
    # Preparing data
    # =================
    # read data
    data_names = list(range(0, 9))
    data = (pd.read_csv("abalone.data", sep=",", header=None, names=data_names))

    # train X, Y
    start_idx = 100
    ys_train = data.iloc[start_idx:, 8]
    xs_train = data.iloc[start_idx:, :8]

    # print("ys_train[0]:: \n", ys_train)
    # print("xs_train[0]:: \n", xs_train)

    # test X, Y
    ys_test = data.iloc[:start_idx, 8]
    xs_test = data.iloc[:start_idx, :8]

    #print(ys_test)
    #print(xs_test)

    # =================
    # Make predictions
    # =================
    good_pred = 0
    # Init array of weights
    w = [Weight() for y in ys_train]
    index = 0
    for obj in w:
        obj.val = 0
        obj.idx = index
        index += 1

    quality = 0
    for test_idx in range(len(ys_test)):
        # Choose from Test set
        ux = xs_test.iloc[test_idx]
        uy = ys_test.iloc[test_idx]

        field_idx = 0
        for u_field in ux:
            #print("calc for field_idx = ", field_idx)
            part_w = dist(xs_train[field_idx], u_field)
            field_idx += 1

            for train_index in range(len(w)):
                w[train_index].val += part_w[train_index]

        # Sorting weights array
        w.sort(key = attrgetter('val'), reverse=True)

        y_pred = 0
        w_sum = 0
        for i in range(k):
            y_pred += ys_train[start_idx + w[i].idx] * w[i].val
            w_sum += w[i].val
        y_pred = round( y_pred / (w_sum))
        quality += ( (y_pred - uy)/max(max(ys_train),max(ys_test)) ) ** 2
        good_pred += 1 if math.fabs(y_pred - uy) <= 1 else 0 # count good predictions
        # print("Predict: ", y_pred,
        #       "\t Reality: ", uy)

    quality = math.sqrt(quality / len(ys_test))
    print("k = ", k, "\t Quality: ", quality)
    return good_pred / len(ys_test)

k_rate = []
ks = []
for k in range(1,10):
    ks.append(k)
    k_rate.append(predicting(k))

for obj in k_rate:
    print(obj)
