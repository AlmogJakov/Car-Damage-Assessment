import math

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

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

n = 1512
for i in range(n):
    # load an image from file
    img_path = f'../database/image_224X224/{i}.jpeg'
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    img = np.array(cv2.imread(img_path, 1) / 255)
    # img = cv2.Canny((img * 255).astype(np.uint8), 125, 175) / 255

    # edges = np.array(cv2.Canny((img * 255).astype(np.uint8), 200, 225) / 255)
    # yiq = transformRGB2YIQ(img)
    # yiq[:, :, 0] += edges * 0.1
    # img = transformYIQ2RGB(yiq)

    # # bilateral
    # img = cv2.imread(img_path)
    # img = cv2.bilateralFilter(img, 50, 25, 50)
    # # LoG
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # sigma = 0.1
    # img = (1-sigma)*img + sigma*cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    # # bilateral + canny
    # img = cv2.imread(img_path)
    # img = cv2.bilateralFilter(img, 50, 25, 50)
    # edges = np.array(cv2.Canny((img * 255).astype(np.uint8), 200, 225) / 255)
    # edges = np.dstack((edges, edges, edges))
    # img = img + edges * img
    # cv2.imwrite(f'database/image_bilateral_224X224/{i}.png', img)

    # # histogram eq
    # img = cv2.imread(img_path)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # # Histogram equalisation on the V-channel
    # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    # # convert image back from HSV to RGB
    # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    # cv2.imwrite(f'database/image_bilateral_224X224/{i}.png', img)

    img = np.array(cv2.imread(img_path))
    (b, g, r) = cv2.split(img)
    r_new = r * 0.393 + g * 0.769 + b * 0.189
    g_new = r * 0.349 + g * 0.686 + b * 0.168
    b_new = r * 0.272 + g * 0.534 + b * 0.131
    img2 = cv2.merge([b_new, g_new, r_new])
    img = np.hstack([
        img,
        img2])
    # img = cv2.bilateralFilter(img, 25, 50, 50)
    # img = white_balance(img)

    # convert img to HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hue, sat, val = cv2.split(hsv)
    # # compute gamma = log(mid*255)/log(mean)
    # mid = 0.5
    # mean = np.mean(val)
    # gamma = math.log(mid * 255) / math.log(mean)
    # print(gamma)
    # # do gamma correction on value channel
    # val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)
    # # combine new value channel with original hue and sat channels
    # hsv_gamma = cv2.merge([hue, sat, val_gamma])
    # img = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

    cv2.imwrite(f'../database/image_bilateral_224X224/{i}.png', img)

    # edges = np.dstack((edges, edges, edges))
    # img = img + (img * edges * 0.2)
    # cv2.imwrite(f'database/image_edge_224X224/{i}.png', (img * 255).astype(np.uint8))
    # plt.imshow(img, cmap='gray')
    # plt.show()



