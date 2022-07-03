import csv
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle


labels_dec = {'unknown': 0, 'head_lamp': 1, 'door_scratch': 2, 'door_dent': 3, 'glass_shatter': 4, 'tail_lamp': 5,
              'bumper_dent': 6, 'bumper_scratch': 7}


def get_csv_labels():
    labels = []
    file = open('database/data.csv')
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
        labels.append(labels_dec[row[2]])
    return np.array(labels)


def split_train_test(vectors_path):
    with open(vectors_path, 'rb') as f:
        data = pickle.load(f)
        data = np.array(data)
    labels = get_csv_labels()
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)
    return X_train, X_test, y_train, y_test


def get_data(project_path):
    data = []
    n = 1512
    for i in range(n):
        img = Image.open(project_path+'/'+str(i)+'.jpg')
        array = np.array(img)
        data.append(array)

    data = np.array(data)
    data = data.reshape((data.shape[0], -1))
    # Normalize
    data = data / 255

    labels = get_csv_labels()

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)
    # print(X_test.shape)
    # exit()
    return X_train, X_test, y_train, y_test