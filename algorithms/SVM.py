# SVM
import numpy as np
from sklearn.svm import SVC


def Support_Vector_Machine(x_train, x_test, y_train, y_test, sample_of_data=False):
    # Run SVM for classification
    model = SVC(decision_function_shape='ovo', C=1, kernel="linear", gamma="auto")
    model.fit(x_train, y_train)
    # Check the accuracy on train and test
    prediction = model.predict(x_train)
    train_acc = 100 * np.sum(prediction == y_train) / len(y_train)
    prediction = model.predict(x_test)
    test_acc = 100 * np.sum(prediction == y_test) / len(y_test)
    print('SVM: train accuracy = {:.2f}%, '
          'test accuracy = {:.2f}%'.format(train_acc, test_acc))
