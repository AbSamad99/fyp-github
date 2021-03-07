import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def word_segment(line):
    # Making a list to return all words in a line
    return_list = []

    # vertical projection profile
    H, W = line.shape[:2]
    th = 2
    hist = cv.reduce(line, 0, cv.REDUCE_AVG).reshape(-1)

    # left and write are the tuples which are for storing the edges of left and right end values
    left = [y for y in range(W-1) if hist[y] <= th and hist[y+1] > th]
    right = [y for y in range(W-1) if hist[y] > th and hist[y+1] <= th]

    transpose = np.transpose(line)

    x = 0
    y = 0
    temp = 1
    while temp < len(left):
        if left[temp]-right[y] > 8:
            return_list.append(np.transpose(
                transpose[left[x]:right[y], :]))
            x = temp
        elif temp == len(left)-1:
            return_list.append(np.transpose(
                transpose[left[x]:right[y+1], :]))
            x = temp
        temp = temp+1
        y = y+1

    return return_list
