from InitialOperations import split_train_test, get_data
from algorithms.CNN import cnn, cnn_via_vgg
# from algorithms.CNN2 import CNN
from algorithms.DTs import decision_tree
from algorithms.SVM import Support_Vector_Machine
from algorithms.kNN import k_nearest_neighbors

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = split_train_test('database/vectors_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/vectors_bilateral_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_bilateralLoG_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_bilateralLoGBlack_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_bilateralLoGCanny_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_histogramEq_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_whiteBalance_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_whiteBalanceGC_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_hsv_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_bgr_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_mean_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_112X112.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_bilateral2_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_blurLvls_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_orgPlusCanny_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_orgPlusBilateral_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_orgPlusflip_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_orgPlusVintage_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_align_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_align45_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_align90_224X224.pkl')
    # x_train, x_test, y_train, y_test = split_train_test('database/image_alignMin_224X224.pkl')

    x_train_img, x_test_img, y_train_img, y_test_img = get_data('database/image_45X45')
    '''
    #####################################################################
    #################### Algorithms Via VGG Vectors #####################
    #####################################################################'''
    '''
     -------------------------------------------------------------------
    |                   CNN + VGG (~70% accuracy)                       |
    |___________________________________________________________________|'''
    cnn_via_vgg(x_train, x_test, y_train, y_test)
    '''
     -------------------------------------------------------------------
    |                   SVM + VGG (~70% accuracy)                       |
    |___________________________________________________________________|'''
    # Support_Vector_Machine(x_train, x_test, y_train, y_test)
    '''
     -------------------------------------------------------------------
    |                   KNN + VGG (~65% accuracy)                       |
    |___________________________________________________________________|'''
    # k_nearest_neighbors(x_train, x_test, y_train, y_test)
    '''
     -------------------------------------------------------------------
    |                   DTs + VGG (~50% accuracy)                       |
    |___________________________________________________________________|'''
    # decision_tree(x_train, x_test, y_train, y_test)

    '''
    #####################################################################
    ################## Algorithms Without VGG Vectors ###################
    #####################################################################'''
    '''
     -------------------------------------------------------------------
    |                      CNN (~55% accuracy)                          |
    |___________________________________________________________________|'''
    # cnn(x_train_img, x_test_img, y_train_img, y_test_img)
    '''
     -------------------------------------------------------------------
    |                      SVM (~70% accuracy)                          |
    |___________________________________________________________________|'''
    #
    '''
     -------------------------------------------------------------------
    |                      KNN (~65% accuracy)                          |
    |___________________________________________________________________|'''
    #
    '''
     -------------------------------------------------------------------
    |                      DTs (~50% accuracy)                          |
    |___________________________________________________________________|'''
    #
