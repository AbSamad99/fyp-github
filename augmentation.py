import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random

folder_path = './Temp/Temp4'
save_path = './Temp/Trials2'

save_size = (75, 75)


def crop_and_pad_by_directory(folder_path):
    for folder in os.listdir(folder_path):
        images = [f for f in os.listdir(folder_path + '/'+folder) if os.path.isfile(
            os.path.join(folder_path+'/' + folder, f))]
        print(folder)
        for image in images:
            img = cv.imread(folder_path+'/' + folder + '/' + image)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.bitwise_not(gray)
            threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

            hist = cv.reduce(threshold, 1, cv.REDUCE_AVG).reshape(-1)
            th = 2
            H, W = img.shape[:2]
            uppers = [y for y in range(
                H-1) if hist[y] <= th and hist[y+1] > th]
            lowers = [y for y in range(
                H-1) if hist[y] > th and hist[y+1] <= th]
            if len(uppers) == 0:
                uppers.append(0)
            if len(lowers) == 0:
                lowers.append(900)

            if len(lowers) == 2:
                img = img[uppers[0]:lowers[1], :]
            else:
                img = img[uppers[0]:lowers[0], :]

            img = cv.rotate(img, cv.cv2.ROTATE_90_CLOCKWISE)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.bitwise_not(gray)
            threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
            hist = cv.reduce(threshold, 1, cv.REDUCE_AVG).reshape(-1)
            th = 2
            H, W = img.shape[:2]
            uppers = [y for y in range(
                H-1) if hist[y] <= th and hist[y+1] > th]
            lowers = [y for y in range(
                H-1) if hist[y] > th and hist[y+1] <= th]
            if len(uppers) == 0:
                uppers.append(0)
            if len(lowers) == 0:
                lowers.append(1200)

            if len(lowers) == 2:
                img = img[uppers[0]:lowers[1], :]
            else:
                img = img[uppers[0]:lowers[0], :]

            img = cv.rotate(img, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

            img = cv.copyMakeBorder(img, 30, 30, 30, 30,
                                    cv.BORDER_CONSTANT, value=[255, 255, 255])

            img = cv.resize(img, (400, 300))

            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # cv.imshow('crop', img)
            # cv.waitKey(0)
            cv.imwrite(folder_path + '/' + folder + '/' + image, img)


def crop_and_pad(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(
        os.path.join(folder_path, f))]
    for image in images:
        img = cv.imread(folder_path + '/' + image)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.bitwise_not(gray)
        threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

        hist = cv.reduce(threshold, 1, cv.REDUCE_AVG).reshape(-1)
        th = 2
        H, W = img.shape[:2]
        uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
        lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
        if len(uppers) == 0:
            uppers.append(0)
        if len(lowers) == 0:
            lowers.append(900)

        if len(lowers) == 2:
            img = img[uppers[0]:lowers[1], :]
        else:
            img = img[uppers[0]:lowers[0], :]

        img = cv.rotate(img, cv.cv2.ROTATE_90_CLOCKWISE)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.bitwise_not(gray)
        threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
        hist = cv.reduce(threshold, 1, cv.REDUCE_AVG).reshape(-1)
        th = 2
        H, W = img.shape[:2]
        uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
        lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
        if len(uppers) == 0:
            uppers.append(0)
        if len(lowers) == 0:
            lowers.append(1200)

        if len(lowers) == 2:
            img = img[uppers[0]:lowers[1], :]
        else:
            img = img[uppers[0]:lowers[0], :]

        img = cv.rotate(img, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

        img = cv.copyMakeBorder(img, 30, 30, 30, 30,
                                cv.BORDER_CONSTANT, value=[255, 255, 255])

        img = cv.resize(img, (400, 300))

        # cv.imshow('crop', img)
        # cv.waitKey(0)
        cv.imwrite(folder_path + '/' + image, img)


def pad_img(img, size):
    img = cv.copyMakeBorder(img, size, size, size, size,
                            cv.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv.resize(img, save_size)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def stretch_horizontal(img, size):
    img = cv.copyMakeBorder(img, size, size, 10, 10,
                            cv.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv.resize(img, save_size)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def stretch_vertical(img, size):
    img = cv.copyMakeBorder(img, 10, 10, size, size,
                            cv.BORDER_CONSTANT, value=[255, 255, 255])
    img = cv.resize(img, save_size)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def rotate_img(img, angle):
    height, width = img.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    img = cv.warpAffine(
        img, rotation_mat, (bound_w, bound_h), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
    img = cv.resize(img, save_size)
    return img


def get_extreme(img, image):
    img = cv.copyMakeBorder(img, 20, 20, 20, 20,
                            cv.BORDER_CONSTANT, value=[255, 255, 255])
    gray = img
    kernel = np.ones((3, 3), np.uint8)
    if len(image) == 11:
        gray = cv.dilate(img, kernel, iterations=2)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

    hist = cv.reduce(threshold, 1, cv.REDUCE_AVG).reshape(-1)
    th = 2
    H, W = img.shape[:2]
    uppers = [y for y in range(
        H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(
        H-1) if hist[y] > th and hist[y+1] <= th]

    if len(uppers) == 0:
        uppers.append(0)
    if len(lowers) == 0:
        lowers.append(200)

    if len(lowers) > 1:
        img = img[uppers[0]:lowers[len(lowers)-1], :]
    else:
        img = img[uppers[0]:lowers[0], :]

    img = cv.rotate(img, cv.cv2.ROTATE_90_CLOCKWISE)
    gray = img
    if len(image) == 11:
        gray = cv.dilate(img, kernel, iterations=2)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    hist = cv.reduce(threshold, 1, cv.REDUCE_AVG).reshape(-1)
    th = 2
    H, W = img.shape[:2]
    uppers = [y for y in range(
        H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(
        H-1) if hist[y] > th and hist[y+1] <= th]
    if len(uppers) == 0:
        uppers.append(0)
    if len(lowers) == 0:
        lowers.append(200)

    if len(lowers) > 1:
        img = img[uppers[0]:lowers[len(lowers)-1], :]
    else:
        img = img[uppers[0]:lowers[0], :]

    img = cv.rotate(img, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def augment(folder_path, save_path):
    for folder in os.listdir(folder_path):
        images = [f for f in os.listdir(folder_path + '/'+folder) if os.path.isfile(
            os.path.join(folder_path+'/' + folder, f))]
        print(folder)
        counter = 1
        for image in images:
            print(image)
            img = cv.imread(folder_path+'/' + folder + '/' + image)

            if not os.path.exists(save_path + '/' + folder):
                os.makedirs(save_path + '/' + folder)

            # saving as is
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", cv.resize(img, save_size))

            counter = counter+1

            # saving extreme crop
            temp = get_extreme(img, image)
            img = cv.resize(temp, save_size)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", img)
            counter = counter+1

            # saving small padded image
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", pad_img(temp, 30))
            counter = counter+1

            # saving large padded image
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", pad_img(temp, 60))
            counter = counter+1

            # saving small horizontally streched image
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", stretch_horizontal(temp, 30))
            counter = counter+1

            # saving small vertically steched image
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", stretch_vertical(temp, 30))
            counter = counter+1

            # saving large horizontally stretched image
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", stretch_horizontal(temp, 60))
            counter = counter+1

            # saving large vertically streched image
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", stretch_vertical(temp, 60))
            counter = counter+1

            # saving image rotated by 20*
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(temp, 20))
            counter = counter+1

            # saving image rotated by -20*
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(temp, -20))
            counter = counter+1

            # saving image rotated by 15*
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(temp, 15))
            counter = counter+1

            # saving image rotated by -15*
            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(temp, -15))
            counter = counter+1

            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(stretch_horizontal(temp, 30), 15))
            counter = counter+1

            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(stretch_vertical(temp, 30), 15))
            counter = counter+1

            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(stretch_horizontal(temp, 30), -15))
            counter = counter+1

            cv.imwrite(save_path + '/' + folder +
                       '/' + str(counter) + ".jpg", rotate_img(stretch_vertical(temp, 30), -15))
            counter = counter+1

            # os.remove(save_path+'/' + folder + '/' + image)
        # break


def salt_and_pepper(img):

    # Getting the dimensions of the image
    row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 1000)
    for i in range(300):

        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 1000)
    for i in range(300):

        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


def noise(folder_path):
    for folder in os.listdir(folder_path):
        images = [f for f in os.listdir(folder_path + '/'+folder) if os.path.isfile(
            os.path.join(folder_path+'/' + folder, f))]
        print(folder)
        counter = len(images)+1
        for image in images:
            img = cv.imread(folder_path+'/' + folder + '/' +
                            image, flags=cv.IMREAD_GRAYSCALE)
            cv.imwrite(folder_path + '/' + folder + '/' +
                       str(counter) + ".jpg", salt_and_pepper(img))
            counter = counter+1

            # cv.imshow('original', img)
            # cv.imshow('salt&pepper', salt_and_pepper(img))
            # cv.waitKey(0)


def rename_folders(folder_path):
    for folder in os.listdir(folder_path):
        new_name = folder.split('img')[1]
        os.rename(folder_path + '/' + folder, folder_path + '/' + new_name)


def renumber_folders(folder_path):
    counter = 1
    for folder in os.listdir(folder_path):
        if counter < 10:
            new_name = '00'+str(counter)
        elif counter < 100:
            new_name = '0'+str(counter)
        else:
            new_name = str(counter)

        print(new_name)
        os.rename(folder_path + '/' + folder, folder_path + '/' + new_name)
        counter = counter + 1


def to_grayscale(folder_path, save_path):
    for folder in os.listdir(folder_path):
        images = [f for f in os.listdir(folder_path + '/'+folder) if os.path.isfile(
            os.path.join(folder_path+'/' + folder, f))]
        print(folder)
        for image in images:
            img = cv.imread(folder_path+'/' + folder + '/' + image)
            img = cv.resize(img, (200, 200))
            # cv.imshow('original', img)
            img = cv.bilateralFilter(img, 10, 20, 20)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # cv.imshow('grayscale', gray)
            threshed = cv.threshold(gray, 90, 255, cv.THRESH_OTSU)[1]
            # cv.imshow('before', threshed)
            count = 0
            if threshed[0][0] == 0:
                count = count+1
            if threshed[0][threshed.shape[1]-1] == 0:
                count = count+1
            if threshed[threshed.shape[0]-1][0] == 0 == 0:
                count = count+1
            if threshed[threshed.shape[0]-1][threshed.shape[1]-1] == 0:
                count = count+1
            if count > 2:
                threshed = cv.bitwise_not(threshed)
            # cv.imshow('after', threshed)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            new_path = save_path + '/' + folder
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            cv.imwrite(new_path + '/' + image, threshed)


def save_one_from_each_class(folder_path, save_path):
    for folder in os.listdir(folder_path):
        images = [f for f in os.listdir(folder_path + '/'+folder) if os.path.isfile(
            os.path.join(folder_path+'/' + folder, f))]
        print(folder)
        counter = 1
        for image in images:
            if image.find('.db') != -1:
                continue
            img = cv.imread(folder_path + '/' + folder + '/' + image)
            cv.imwrite(save_path + '/' + image, img)
            break


def save_all_from_each_class(folder_path, save_path):
    for folder in os.listdir(folder_path):
        images = [f for f in os.listdir(folder_path + '/'+folder) if os.path.isfile(
            os.path.join(folder_path+'/' + folder, f))]
        print(folder)
        counter = 1
        for image in images:
            if image.find('.db') != -1:
                continue
            img = cv.imread(folder_path + '/' + folder + '/' + image)
            cv.imwrite(save_path + '/' + image, img)


augment(folder_path, save_path)

# rename_folders(folder_path)

# renumber_folders(folder_path)

# noise(folder_path)

# save_one_from_each_class(folder_path, save_path)

# to_grayscale(folder_path, save_path)
