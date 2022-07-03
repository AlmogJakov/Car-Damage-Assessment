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


<<<<<<< HEAD
def cnn(X_train, X_test, y_train, y_test):
    # ROW = 45
    # COL = 45
    # CHANNEL = 3
    # X_train = X_train.reshape(X_train.shape[0], ROW, COL, CHANNEL)
    # X_test = X_test.reshape(X_test.shape[0], ROW, COL, CHANNEL)
=======
def CNN(X_train, X_test, y_train, y_test):
    ROW = 45
    COL = 20
    CHANNEL = 1

    # X_train = X_train.reshape(X_train.shape[0], ROW, COL, CHANNEL)
    # X_test = X_test.reshape(X_test.shape[0], ROW, COL, CHANNEL)
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
>>>>>>> 89bbd590ccc5a9bdf6c730f1df1a3bfb62fd8c36

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
<<<<<<< HEAD
    results = model.evaluate(X_test, y_test, verbose=0)
    print("final accuracy: {:5.2f}%".format(100 * results[1]))
=======


def CNN2(X_train, y_train, X_test, y_test):
    print(X_train.shape)
    print(y_train.shape)
    verbose, epochs, batch_size = 0, 10, 32
    print(X_train.shape)
    n_outputs = y_train.shape[0]
    y_train = y_train.reshape(1209, )
    # n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[0]
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(accuracy*100)
    # return accuracy
>>>>>>> 89bbd590ccc5a9bdf6c730f1df1a3bfb62fd8c36
