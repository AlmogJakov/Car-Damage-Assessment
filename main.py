from InitialOperations import split2_train_test
from algorithms.CNN import CNN
from algorithms.SVM import Support_Vector_Machine
from algorithms.kNN import k_nearest_neighbors

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = split2_train_test()

    # CNN
    # CNN(x_train, x_test, y_train, y_test)

    # PCA
    # x_train_2D, x_test_2D, y_train, y_test = Principal_Component_Analysis(x_train, x_test, y_train, y_test, True, False)

    # SVM
    Support_Vector_Machine(x_train, x_test, y_train, y_test)

    # SVM after PCA
    # Support_Vector_Machine(x_train_2D, x_test_2D, y_train, y_test)

    # KNN
    # k_nearest_neighbors(x_train, x_test, y_train, y_test)

    # KNN after PCA
    # k_nearest_neighbors(x_train_2D, x_test_2D, y_train, y_test)



