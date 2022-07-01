from keras import Sequential
from keras.callbacks import LambdaCallback
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Conv2D, MaxPool2D
from matplotlib import pyplot as plt


def evaluate_test(model, X_test, y_test, test_acc, test_loss):
    res_test = model.evaluate(X_test, y_test)
    test_loss.append(res_test[0])
    test_acc.append(res_test[1])


def CNN(X_train, X_test, y_train, y_test):
    ROW = 100
    COL = 10
    CHANNEL = 1

    X_train = X_train.reshape(X_train.shape[0], ROW, COL, CHANNEL)
    X_test = X_test.reshape(X_test.shape[0], ROW, COL, CHANNEL)
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(32))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    epochs = 6
    test_error = []
    test_acc = []
    myCallback = LambdaCallback(on_epoch_end=lambda batch, logs: evaluate_test(model, X_test, y_test, test_acc, test_error))
    details = model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_split=0.214, callbacks=[myCallback])

    val_acc_list = details.history['val_accuracy']
    acc_list = details.history['accuracy']
    loss_list = details.history['loss']
    val_loss_list = details.history['val_loss']
    results = model.evaluate(X_test, y_test, verbose=0)

    print("final accuracy: {:5.2f}%".format(100 * results[1]))

    x_axis = []
    k = 1
    for i in range(1, epochs + 1):
        x_axis.append(k)
        k = k + 1

    # Loss graph
    plt.plot(x_axis, test_error, label="test loss")
    plt.plot(x_axis, val_loss_list, label="validation loss")
    plt.plot(x_axis, loss_list, label="train loss")
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('train and validation loss')
    plt.legend()
    plt.show()

    # Accuracy graph
    plt.plot(x_axis, test_acc, label="test accuracy")
    plt.plot(x_axis, val_acc_list, label="validation accuracy")
    plt.plot(x_axis, acc_list, label="train accuracy")
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('train and validation accuracy')
    plt.legend()
    plt.show()
