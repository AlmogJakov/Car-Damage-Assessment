import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

database = 'database/'
csv_file = 'database/data.csv'
img_dir = 'database/image_resized2'
dirs_resize = os.listdir(database + img_dir)


def split2_train_test():
    data = []
    labels = []
    details = pd.read_csv(database + csv_file)
    images_names = details.iloc[:, 0]
    images_names = images_names.values.tolist()
    images_labels = details.iloc[:, 3]
    images_labels = images_labels.values.tolist()
    kids_clothes = images_labels.count(1)
    for item in dirs_resize:
        f, e = os.path.splitext(item)
        if f in images_names:
            index = images_names.index(f)
            label = images_labels[index]
            img = Image.open(database+img_dir+'/'+item)
            array = np.array(img)
            if label == 1:
                data.append(array)
                labels.append(label)
            elif kids_clothes > 0:
                data.append(array)
                labels.append(label)
                kids_clothes = kids_clothes - 1

    data = np.array(data)
    data = data.reshape((data.shape[0], -1))
    # Normalize
    data = data / 255
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test