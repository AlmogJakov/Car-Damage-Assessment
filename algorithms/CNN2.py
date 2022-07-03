# importing the required libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

def CNN(X_train, X_test, y_train, y_test):
    # loading data
    # (X_train, y_train), (X_test, y_test) = (X_train, y_train), (X_test, y_test)
    # reshaping data
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    ROW = 100
    COL = 10
    CHANNEL = 1

    X_train = X_train.reshape(X_train.shape[0], ROW, COL, CHANNEL)
    X_test = X_test.reshape(X_test.shape[0], ROW, COL, CHANNEL)

    # checking the shape after reshaping
    print(X_train.shape)
    print(X_test.shape)
    # normalizing the pixel values
    X_train = X_train / 255
    X_test = X_test / 255

    # defining model
    model = Sequential()
    # adding convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # adding pooling layer
    model.add(MaxPool2D(2, 2))
    # adding fully connected layer
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # adding output layer
    model.add(Dense(10, activation='softmax'))
    # compiling the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fitting the model
    model.fit(X_train, y_train, epochs=10)

