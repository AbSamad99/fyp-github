# line_segmentation>

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def line_segmentation(threshed_img, original_img):
    # Making a list to store and return all lines
    return_list = []

    # Get the histogram
    hist = cv.reduce(threshed_img, 1, cv.REDUCE_AVG).reshape(-1)

    # Storing each line in the list
    start = 0
    for i in range(threshed_img.shape[0]):
        if threshed_img[i].any() == 0:
            if i-start > 10:
                return_list.append(original_img[start:i+2])
            start = i

    return return_list
