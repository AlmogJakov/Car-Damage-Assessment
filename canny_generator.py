import cv2
import numpy as np
import matplotlib.pyplot as plt


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    try:
        backup_shape = imgRGB.shape
        res = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(backup_shape)
        # We need to multiply by 255.0 and then round to the nearest Integer number (with 'np.rint' func) and then
        # normalize again (divide by 255.0) in order to allow the image to contain a completely white color (255)
        return res
    except (Exception,):
        print("An exception occurred: can't convert the image from RGB to YIQ")
        exit(1)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    try:
        backup_shape = imgYIQ.shape
        res = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(backup_shape)
        # We need to multiply by 255.0 and then round to the nearest Integer number (with 'np.rint' func) and then
        # normalize again (divide by 255.0) in order to allow the image to contain a completely white color (255)
        return np.clip(res, 0, 1)
    except (Exception,):
        print("An exception occurred: can't convert the image from RGB to YIQ")
        exit(1)


n = 1512
for i in range(n):
    # load an image from file
    img_path = f'database/image_224X224/{i}.jpeg'
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    img = np.array(cv2.imread(img_path, 1) / 255)
    # img = cv2.Canny((img * 255).astype(np.uint8), 125, 175) / 255

    # edges = np.array(cv2.Canny((img * 255).astype(np.uint8), 200, 225) / 255)
    # yiq = transformRGB2YIQ(img)
    # yiq[:, :, 0] += edges * 0.1
    # img = transformYIQ2RGB(yiq)

    img = cv2.imread(img_path)
    img = cv2.bilateralFilter(img, 50, 25, 50)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sigma = 0.1
    img = (1-sigma)*img + sigma*cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    cv2.imwrite(f'database/image_bilateral_224X224/{i}.png', img)

    # edges = np.dstack((edges, edges, edges))
    # img = img + (img * edges * 0.2)
    # cv2.imwrite(f'database/image_edge_224X224/{i}.png', (img * 255).astype(np.uint8))
    # plt.imshow(img, cmap='gray')
    # plt.show()
