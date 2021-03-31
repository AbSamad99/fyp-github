import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def skew_correction(image):
    # convert the image to grayscale and threshold
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    threshold = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coordinates = np.column_stack(np.where(threshold > 0))
    angle = cv.minAreaRect(coordinates)[-1]

    rotated = img

    # the `cv.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < 89:
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90-angle
    # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(
        img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated
