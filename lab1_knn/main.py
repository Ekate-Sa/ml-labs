import pandas as pd
import numpy as np
import math
from operator import attrgetter

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, p=2)


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

def maximum(a):
    max_num  = a[0]
    max_i = 0
    for i in range(1,len(a)):
        if a[i]>max_num:
            max_i = i
            max_num = a[i]
    return max_i

def knn_predict(xs_test, xs_train, ys_train, k):
    # =================
    # Make predictions
    # =================

    # Init array of weights
    w = [Weight() for y in ys_train[8]]
    index = 0
    for obj in w:
        obj.val = 0
        obj.idx = index
        index += 1

    quality = 0
    # Predictions array
    ys_pred = np.zeros((len(xs_test),) , dtype=int)
    for test_idx in range(len(xs_test)):
        # Choose from Test set
        ux = xs_test.iloc[test_idx]
        field_idx = 0
        for u_field in ux:
            # print("calc for field_idx = ", field_idx)
            part_w = dist(xs_train[field_idx], u_field)
            field_idx += 1

            for train_index in range(len(w)):
                w[train_index].val += part_w[train_index]

        #
        # Sorting weights array
        #
        w.sort(key = attrgetter('val'), reverse=True)


        # Create array to count Y values in kNN
        y_spectre = np.zeros(2*max(ys_train[8]) - min(ys_train[8]))

        for i in range(0,k):
            y_spectre[ys_train.iloc[w[i].idx]] += 1 # Here -1

        ys_pred[test_idx] = maximum(y_spectre) # Here +1
        # if (test_idx == 0):
        #     print("k=", k, "Y SPECTRE : ", y_spectre)

    return pd.DataFrame(data = ys_pred)

def calc_quality(ys_pred, ys_test):
    quality = 0
    for pred, train in zip(ys_pred[0], ys_train[8]):
        quality += (pred - train) ** 2

    quality = quality / len(ys_pred[0])
    return quality

def calc_recall(ys_pred, ys_test):
    trues = 0;
    for idx in ys_pred:
        if (ys_pred[idx] == ys_test[idx]):
            trues += 1
    recall = trues / len(ys_pred)
    return recall

# =================
# Preparing data
# =================
# read data
data_names = list(range(0, 9))
data = (pd.read_csv("abalone.data", sep=",", header=None, names=data_names))

data_test = data.sample(frac=0.2)
data_train = data.drop(data_test.index)

ys_test = data_test.iloc[0:,  8].reset_index()      # Needed for Quality measurement
xs_test = data_test.iloc[0:, :8].reset_index()
ys_train = data_train.iloc[0:,  8].reset_index()
xs_train = data_train.iloc[0:, :8].reset_index()

# for test_item in ys_test["index"]:
#     for train_item in ys_train["index"]:
#         if (test_item == train_item):
#             print("two similar found")

del ys_test["index"]
del xs_test["index"]
del ys_train["index"]
del xs_train["index"]
# print("X TEST \n", xs_test,
#       "Y TEST \n", ys_test)

#
# Call kNN prediction
#

#=====================
# Start here
#====================

for k in range(1,5):
    ys_pred = knn_predict(xs_test, xs_train, ys_train, k)
    recall = calc_recall(ys_pred[0], ys_test[8])
    # print(ys_pred,"\n", ys_test)
    print("k = ", k, " Recall = ", recall)

# ys_pred_sk = []
# for elem in xs_test:
#     ys_pred_sk.append(neigh.predict(elem))
#     recall = calc_recall(ys_pred_sk, ys_test[8])
#     print("SK:: k = ", k, " Recall = ", recall)

