import numpy as np
from keras import Sequential
from keras.callbacks import LambdaCallback
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Conv2D, MaxPool2D, Conv1D, Dropout, MaxPooling1D
from matplotlib import pyplot as plt
from tensorflow import keras


# https://www.kaggle.com/code/anandhuh/image-classification-using-cnn-for-beginners/notebook

def evaluate_test(model, X_test, y_test, test_acc, test_loss):
    res_test = model.evaluate(X_test, y_test)
    test_loss.append(res_test[0])
    test_acc.append(res_test[1])


def cnn(X_train, X_test, y_train, y_test):
    # ROW = 45
    # COL = 45
    # CHANNEL = 3
    # X_train = X_train.reshape(X_train.shape[0], ROW, COL, CHANNEL)
    # X_test = X_test.reshape(X_test.shape[0], ROW, COL, CHANNEL)

    # model = Sequential()
    # model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model = Sequential()
    model.add(Conv2D(45, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(62, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(62, (3, 3), activation='relu'))

    model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
    model.add(Dense(62, activation='relu'))
    model.add(Dense(32))

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.25, 1])
    plt.legend(loc='lower right')
    plt.show()

    results = model.evaluate(X_test, y_test, verbose=0)
    print("final accuracy: {:5.2f}%".format(100 * results[1]))


def cnn_via_vgg(X_train, X_test, y_train, y_test):
    model = Sequential()
    # model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
    model.add(Dense(62, activation='relu'))
    model.add(Dense(32))
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.25, 1])
    plt.legend(loc='lower right')
    plt.show()
    results = model.evaluate(X_test, y_test, verbose=0)
    print("final accuracy: {:5.2f}%".format(100 * results[1]))
