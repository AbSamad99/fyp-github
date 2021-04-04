import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def skew_correction(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

    coordinates = np.column_stack(np.where(threshold > 0))
    angle = cv.minAreaRect(coordinates)[-1]

    rotated = img

    if angle < 89:
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90-angle
        else:
            angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(
        img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    return rotated
