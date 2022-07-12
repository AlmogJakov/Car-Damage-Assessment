# SVM
import numpy as np
from sklearn.svm import SVC


def Support_Vector_Machine(x_train, x_test, y_train, y_test, sample_of_data=False):
    ##########################
    # if sample_of_data is true we sample 10% from the hole data
    # if sample_of_data is True:
    #     num_train = len(x_train) // 10
    #     num_test = len(x_test) // 10
    #     train_indices = np.random.randint(0, len(x_train), num_train)
    #     test_indices = np.random.randint(0, len(x_test), num_test)
    #
    #     x_train = x_train[train_indices, :]
    #     y_train = y_train[train_indices]
    #     x_test = x_test[test_indices, :]
    #     y_test = y_test[test_indices]
    #############################################
    # Run SVM for classification
    model = SVC(C=1, kernel="linear", gamma="auto")
    model.fit(x_train, y_train)
    # Check the accuracy on train and test
    prediction = model.predict(x_train)
    train_acc = 100 * np.sum(prediction == y_train) / len(y_train)
    prediction = model.predict(x_test)
    test_acc = 100 * np.sum(prediction == y_test) / len(y_test)
    print('SVM: train accuracy = {:.2f}%, '
          'test accuracy = {:.2f}%'.format(train_acc, test_acc))
