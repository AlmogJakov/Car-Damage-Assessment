from InitialOperations import split_train_test, get_data
from algorithms.CNN import *
from algorithms.DTs import decision_tree
from algorithms.SVM import Support_Vector_Machine
from algorithms.kNN import k_nearest_neighbors


def database_tweak_switch():
    switcher = {
        1: ["Original", 'database/vectors_224X224.pkl'],
        2: ["Bilateral filter", 'database/vectors_bilateral_224X224.pkl'],
        3: ["Bilateral filter + LoG", 'database/image_bilateralLoG_224X224.pkl'],
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
        19: ["Alignment Via Hough Line Transform (90 Degree)", 'database/image_align_224X224.pkl'],
        20: ["Alignment Via Hough Line Transform (45 Degree)", 'database/image_align45_224X224.pkl'],
        21: ["Alignment Via Hough Line Transform (90 Degree)", 'database/image_align90_224X224.pkl'],
        22: ["Alignment Via Hough Line Transform (Minimum Degree)", 'database/image_alignMin_224X224.pkl']
    }
    print("Database tweaks:")
    max_desc = 0
    for key, value in switcher.items():
        if len(value[0]) > max_desc:
            max_desc = len(value[0])
    print("   +----+" + "-" * (max_desc + 3) + "+")
    for key, value in switcher.items():
        print("   |", f'{key:<2}', '| ', f'{value[0]:<{max_desc + 1}}' + "|")
        print("   +----+" + "-" * (max_desc + 3) + "+")
    option = int(input("Please choose option: "))
    err_msg = "Invalid input"
    if switcher.get(option, err_msg) == err_msg:
        return None
    return switcher.get(option, "Invalid input")[1]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    vgg_option = ""
    while True:
        vgg_option = input(f"{bcolors.BOLD}Use VGG? (Y/N): {bcolors.ENDC}")
        if vgg_option == "Y" or vgg_option == "y" or vgg_option == "N" or vgg_option == "n":
            break
    options = {"cnn": "CNN", "svm": "SVM", "knn": "KNN", "dts": "DTs", "db_tweaks": "Database Tweak"}
    max_desc = 0
    for key, value in options.items():
        if len(value) > max_desc:
            max_desc = len(value)
    if vgg_option == "Y" or vgg_option == "y":
        # Init with default database
        vgg_database_name = 'database/vectors_224X224.pkl'
        while True:
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 1  | " + f'{options["cnn"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 2  | " + f'{options["svm"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 3  | " + f'{options["knn"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 4  | " + f'{options["dts"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 5  | " + f'{options["db_tweaks"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            option = input(f"{bcolors.BOLD}Please choose option: {bcolors.ENDC}")
            x_train, x_test, y_train, y_test = split_train_test(vgg_database_name)
            if option == "1":
                cnn_via_vgg(x_train, x_test, y_train, y_test)
                break
            elif option == "2":
                Support_Vector_Machine(x_train, x_test, y_train, y_test)
                break
            elif option == "3":
                k_nearest_neighbors(x_train, x_test, y_train, y_test)
                break
            elif option == "4":
                decision_tree(x_train, x_test, y_train, y_test)
                break
            elif option == "5":
                print()
                res = database_tweak_switch()
                if res is None:
                    print(f"{bcolors.WARNING}Invalid Input{bcolors.ENDC}")
                    print()
                else:
                    vgg_database_name = res

    else:
        while True:
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 1  | " + f'{options["cnn"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 2  | " + f'{options["svm"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 3  | " + f'{options["knn"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            print("   | 4  | " + f'{options["dts"]:<{max_desc + 1}}' + "|")
            print("   +----+" + "-" * (max_desc + 2) + "+")
            option = input(f"{bcolors.BOLD}Please choose option: {bcolors.ENDC}")
            x_train_img, x_test_img, y_train_img, y_test_img = get_data('database/image_45X45')
            if option == "1":
                cnn(x_train_img, x_test_img, y_train_img, y_test_img)
                break
            elif option == "2":
                Support_Vector_Machine(x_train_img, x_test_img, y_train_img, y_test_img)
                break
            elif option == "3":
                k_nearest_neighbors(x_train_img, x_test_img, y_train_img, y_test_img)
                break
            elif option == "4":
                decision_tree(x_train_img, x_test_img, y_train_img, y_test_img)
                break

    '''
    #####################################################################
    #################### Algorithms Via VGG Vectors #####################
    #####################################################################'''
    '''
     -------------------------------------------------------------------
    |                   CNN + VGG (~70% accuracy)                       |
    |___________________________________________________________________|'''
    # cnn_via_vgg(x_train, x_test, y_train, y_test)
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
    |                      SVM (~30% accuracy)                          |
    |___________________________________________________________________|'''
    # Support_Vector_Machine(x_train_img, x_test_img, y_train_img, y_test_img)
    '''
     -------------------------------------------------------------------
    |                      KNN (~35% accuracy)                          |
    |___________________________________________________________________|'''
    # k_nearest_neighbors(x_train_img, x_test_img, y_train_img, y_test_img)
    '''
     -------------------------------------------------------------------
    |                      DTs (~35% accuracy)                          |
    |___________________________________________________________________|'''
    # decision_tree(x_train_img, x_test_img, y_train_img, y_test_img)
