import csv
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle


labels_dec = {'unknown': 0, 'head_lamp': 1, 'door_scratch': 2, 'door_dent': 3, 'glass_shatter': 4, 'tail_lamp': 5,
              'bumper_dent': 6, 'bumper_scratch': 7}


def split_train_test(vectors_path):
    labels = []
    with open(vectors_path, 'rb') as f:
        data = pickle.load(f)
    file = open('database/data.csv')
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        labels.append(labels_dec[row[2]])
    data = np.array(data)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)
    # print(type(X_test))
    # print(X_test.shape)
    #
    # exit()
    return X_train, X_test, y_train, y_test
