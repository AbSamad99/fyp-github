import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def word_segment(line_threshed, line_original):
    # Making a list to return all words in a line_threshed
    return_list = []

    # vertical projection profile
    H, W = line_threshed.shape[:2]
    th = 2
    hist = cv.reduce(line_threshed, 0, cv.REDUCE_AVG).reshape(-1)

    # left and write are the tuples which are for storing the edges of left and right end values
    left = [y for y in range(W-1) if hist[y] <= th and hist[y+1] > th]
    right = [y for y in range(W-1) if hist[y] > th and hist[y+1] <= th]

    transpose = np.transpose(line_original)

    x = 0
    y = 0
    temp = 1
    while temp < len(left):
        # print(left[temp]-right[y], right[y+1]-right[y])
        if left[temp]-right[y] > 5 and right[y+1]-right[y] > 45:
            return_list.append(np.transpose(
                transpose[left[x]:right[y]+2]))
            x = temp
        if temp == len(left)-1:
            return_list.append(np.transpose(
                transpose[left[x]:right[y+1]+2]))
            x = temp
        temp = temp+1
        y = y+1

    return return_list
