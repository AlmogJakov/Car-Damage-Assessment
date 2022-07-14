# KNN
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    neighbors = np.arange(1, 25)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    times = 5
    for j in range(times):
        # Loop over K values
        for i, k in enumerate(neighbors):
            knn = KNeighborsClassifier(n_neighbors=k)  # leaf_size=4
            knn.fit(X_train, y_train)

            # Compute training and test data accuracy
            train_accuracy[i] += knn.score(X_train, y_train)
            test_accuracy[i] += knn.score(X_test, y_test)

    train_accuracy /= times
    test_accuracy /= times
    # Generate plot
    plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()