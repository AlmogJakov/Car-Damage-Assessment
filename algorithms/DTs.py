# DTs
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree


def decision_tree_classifier(X_train, X_test, y_train, y_test):
    depth = np.arange(1, 10)
    train_accuracy = np.empty(len(depth))
    test_accuracy = np.empty(len(depth))

    # Loop over K depth
    for i, k in enumerate(depth):
        dtc = tree.DecisionTreeClassifier(random_state=0, max_depth=k)
        dtc.fit(X_train, y_train)

        # Compute training and test data accuracy
        train_accuracy[i] = dtc.score(X_train, y_train)
        test_accuracy[i] = dtc.score(X_test, y_test)

    # Generate plot
    plt.plot(depth, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(depth, train_accuracy, label='Training dataset Accuracy')

    plt.legend()
    plt.xlabel('depth')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()


def decision_tree(X_train, X_test, y_train, y_test):
    decision_tree_classifier(X_train, X_test, y_train, y_test)
