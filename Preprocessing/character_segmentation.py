import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def character_segment(word):
    # Creating a list to return all characters
    return_list = []

    # Padding the word for better detection
    word = cv.copyMakeBorder(word, 5, 5, 5, 5, borderType=cv.BORDER_CONSTANT)
    cv.imshow('padded', word)

    # vertical projection profile
    H, W = word.shape[:2]
    th = 2
    hist = cv.reduce(word, 0, cv.REDUCE_AVG).reshape(-1)

    # left and write are the tuples which are for storing the edges of left and right end values
    left = [y for y in range(W-1) if hist[y] <= th and hist[y+1] > th]
    right = [y for y in range(W-1) if hist[y] > th and hist[y+1] <= th]

    for i in range(len(left)):
        return_list.append(np.transpose(transpose[left[i]:right[i]]))

    return return_list
