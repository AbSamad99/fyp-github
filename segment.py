import os
import cv2 as cv
from Preprocessing.line_segmentation import line_segmentation
from Preprocessing.word_segmentation import word_segment
from Preprocessing.character_segmentation import character_segment

load_path = './Typed Crops/img648-img657 (img646_647 skipped)'
save_path = './Temp/Trials3'


def segment(load_path):
    arr = []
    for i in range(648, 658):
        if i == 433:
            continue
        if i < 10:
            arr.append('img00'+str(i))
        elif i < 100:
            arr.append('img0'+str(i))
        else:
            arr.append('img'+str(i))
    print(arr)

    i = 1

    images = [f for f in os.listdir(load_path) if os.path.isfile(
        os.path.join(load_path, f))]

    for image in images:
        img = cv.imread(load_path + '/' + image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cv.imshow('gray', gray)
        thresh = cv.threshold(gray, 250, 255,  cv.THRESH_BINARY_INV)[1]
        # cv.imshow('thresh', thresh)
        # cv.waitKey(0)
        lines = line_segmentation(thresh, gray)

        for line in lines:
            # cv.imshow('line', line)
            # cv.waitKey(0)
            line_thresh = cv.threshold(
                line, 250, 255, cv.THRESH_BINARY_INV)[1]
            words = word_segment(line_thresh, line)

            print('length', len(words))

            j = 0

            for word in words:
                if len(words) != len(arr):
                    # cv.imshow('word', word)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    break
                # word = cv.resize(word, (200, 200))
                # if not os.path.exists(save_path+'/'+arr[j]):
                #     os.makedirs(save_path+'/'+arr[j])
                cv.imwrite(save_path + '/' +
                           arr[j] + '/' + str(i) + '.png', word)
                j = j+1

            cv.destroyAllWindows()

            i = i+1


segment(load_path)
