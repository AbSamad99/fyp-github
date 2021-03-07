import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def line_segmentation(threshed_img):
    # Making a list to store and return all lines
    return_list = []

    # Get the histogram
    hist = cv.reduce(threshed_img, 1, cv.REDUCE_AVG).reshape(-1)

    # Getting the upper and lower values of pixel for segmentation
    th = 2
    H, W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]

    # Storing each line in the list
    start = 0
    for i in range(threshed_img.shape[0]):
        if threshed_img[i].any() == 0:
            if i-start > 10:
                return_list.append(threshed_img[start:i, :])
            start = i

    return return_list
