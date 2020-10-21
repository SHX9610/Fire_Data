import numpy as np
import cv2


def createCheckerBoard(img1, img2, stride=8):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    rows, cols, channels = image1.shape
    width_pad = int(cols / stride)
    hight_pad = int(rows / stride)
    for i in range(0, stride, 2):
        for j in range(0, stride, 2):
            image2[i * width_pad:(i + 1) * width_pad, j * hight_pad:(j + 1) * hight_pad, :] = image1[i * width_pad:(i + 1) * width_pad,j * hight_pad:(j + 1) * hight_pad,:]
    for i in range(1, stride, 2):
        for j in range(1, stride, 2):
            image2[i * width_pad:(i + 1) * width_pad, j * hight_pad:(j + 1) * hight_pad, :] = image1[i * width_pad:(i + 1) * width_pad,j * hight_pad:(j + 1) * hight_pad,:]
    return image2