# import matplotlib.pyplot as plt
# plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from InitialOperations import split_train_test


def database_tweak_switch(op, idx):
    switcher = {
        1: ["Original", 'database/vectors_224X224.pkl'],
        2: ["Bilateral filter", 'database/vectors_bilateral_224X224.pkl'],  # ~65-70%
        3: ["Bilateral filter + LoG", 'database/image_bilateralLoG_224X224.pkl'],  # ~70%
        4: ["Bilateral filter + LoG (Black & White)", 'database/image_bilateralLoGBlack_224X224.pkl'],
        5: ["Bilateral filter + LoG + Canny", 'database/image_bilateralLoGCanny_224X224.pkl'],
        6: ["Histogram Equalization", 'database/image_histogramEq_224X224.pkl'],
        7: ["White Balance", 'database/image_whiteBalance_224X224.pkl'],
        8: ["White Balance + Gamma Correction", 'database/image_whiteBalanceGC_224X224.pkl'],
        9: ["HSV", 'database/image_hsv_224X224.pkl'],
        10: ["BGR", 'database/image_bgr_224X224.pkl'],
        11: ["Mean reduction", 'database/image_mean_224X224.pkl'],
        12: ["Resized Images (112X112)", 'database/image_112X112.pkl'],
        13: ["Bilateral filter (2)", 'database/image_bilateral2_224X224.pkl'],
        14: ["Stacked images (3 Levels of Blur)", 'database/image_blurLvls_224X224.pkl'],
        15: ["Stacked images (Original + Canny)", 'database/image_orgPlusCanny_224X224.pkl'],
        16: ["Stacked images (Original + Bilateral)", 'database/image_orgPlusBilateral_224X224.pkl'],
        17: ["Stacked images (Original + Flipped)", 'database/image_orgPlusflip_224X224.pkl'],
        18: ["Stacked images (Original + Vintage)", 'database/image_orgPlusVintage_224X224.pkl'],
        19: ["Alignment Via HLT (90 Deg)", 'database/image_align_224X224.pkl'],
        20: ["Alignment Via HLT (45 Deg)", 'database/image_align45_224X224.pkl'],
        21: ["Alignment Via HLT (90 Deg)", 'database/image_align90_224X224.pkl'],
        22: ["Alignment Via HLT (Min Deg)", 'database/image_alignMin_224X224.pkl']
    }
    return switcher.get(op, "Invalid input")[idx]


def SVM(x_train, x_test, y_train, y_test):  # test accuracy only
    # Run SVM for classification
    model = SVC(C=1, kernel="linear", gamma="auto")
    model.fit(x_train, y_train)
    # Check the accuracy on train and test
    prediction = model.predict(x_test)
    test_acc = 100 * np.sum(prediction == y_test) / len(y_test)
    return test_acc

objects = []
performance = []
objects_num = 22
for i in range(1, objects_num + 1):
    objects.append(database_tweak_switch(i, 0))
    vgg_database_name = database_tweak_switch(i, 1)
    x_train, x_test, y_train, y_test = split_train_test(vgg_database_name)
    res = SVM(x_train, x_test, y_train, y_test)
    performance.append(float("{:.2f}".format(res)))
# Source code: https://pythonspot.com/matplotlib-bar-chart/
y_pos = np.arange(len(objects))
plt.figure(figsize=(8, 4))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.barh(y_pos, performance, align='center', alpha=1.0, color=colors)
plt.yticks(y_pos, objects)
plt.xlim(0, np.max(performance) * 1.4)
# plt.xlabel('Usage')
plt.title('Comparison results')
# for i, v in enumerate(performance):
#     plt.text(v + 0.5, i, str(v), color='blue', fontweight='bold')
plt.tight_layout()
plt.show()
