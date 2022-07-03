import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel
from matplotlib import cm
import cv2


def binarizeImage(RGB_image):
    image = rgb2gray(RGB_image)
    threshold = threshold_otsu(image)
    bina_image = image < threshold
    return bina_image


def findEdges(bina_image):
    image_edges = sobel(bina_image)
    return image_edges


def findTiltAngle(image_edges):
    h, theta, d = hough_line(image_edges)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    # angle = np.rad2deg(mode(angles)[0][0])
    bestH, bestTheta, bestD = skimage.transform.hough_line_peaks(h, theta, d)
    angle = np.array(bestTheta).min() * (180 / np.pi)
    if angle < 0:
        angle = angle + 90
    else:
        angle = angle - 90
    return angle


def rotateImage(RGB_image, angle):
    return rotate(RGB_image, angle)

n = 1512
for i in range(n):
    # load an image from file
    img_path = f'database/image_224X224/{i}.jpeg'
    img = np.array(cv2.imread(img_path))
    # print(findTiltAngle(findEdges(binarizeImage(img))))
    img = rotateImage(img, findTiltAngle(findEdges(binarizeImage(img)))) * 255
    cv2.imwrite(f'database/image_bilateral_224X224/{i}.png', img)