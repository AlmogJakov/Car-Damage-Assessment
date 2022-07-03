from InitialOperations import split_train_test
from algorithms.CNN import CNN
# from algorithms.CNN2 import CNN
from algorithms.DTs import decision_tree
from algorithms.SVM import Support_Vector_Machine
from algorithms.kNN import k_nearest_neighbors

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = split_train_test('database/vectors_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/vectors_bilateral_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_bilateralLoG_224X224.pkl')

    # CNN
    # CNN(x_train, x_test, y_train, y_test)

    # SVM
    Support_Vector_Machine(x_train, x_test, y_train, y_test)

    # KNN
    # k_nearest_neighbors(x_train, x_test, y_train, y_test)

    # DTs
    # decision_tree(x_train, x_test, y_train, y_test)




