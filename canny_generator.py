import cv2
import numpy as np
import matplotlib.pyplot as plt

n = 1512
for i in range(n):
    # load an image from file
    img_path = f'database/image_224X224/{i}.jpeg'
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
    img = cv2.Canny((img * 255).astype(np.uint8), 125, 175) / 255
    cv2.imwrite(f'database/image_canny_224X224/{i}.png', (img * 255).astype(np.uint8))
    # plt.imshow(img, cmap='gray')
    # plt.show()
