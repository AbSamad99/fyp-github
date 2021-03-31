# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import numpy as np

import os
import os.path
import shutil

folder_path = "./Temp/Temp2"
save_path = './Temp/Img'


def sort_dataset(folder_path, save_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(
        os.path.join(folder_path, f))]

    for image in images:
        folder_name = image.split('-')[0]

        # print(folder_name)

        new_path = os.path.join(save_path, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        old_image_path = os.path.join(folder_path, image)
        new_image_path = os.path.join(new_path, image)
        shutil.copyfile(old_image_path, new_image_path)


sort_dataset(folder_path, save_path)

# df = pd.read_csv(
#     "./Kannada_Images/kannada.csv", dtype={"img": pd.StringDtype()}
# )

# df["class"] = df["class"].astype("category")

# values = list(df.columns.values)

# print(values)

# y = df[values[-1:]]
# y = np.array(y, dtype='int')
# print(y)

# print(df['class'])

# for i in range(16424, 16425):
#     img = cv.cvtColor(cv.imread("./Kannada_Images/" +
#                                 df.at[i, "img"]), cv.COLOR_BGR2GRAY)
#     cv.imwrite('./Temp/Second/'+df.at[i, "img"], img)

# img = cv.imread("./Temp/" + df.at[10, "img"], cv.IMREAD_GRAYSCALE)

# print(img.shape)

# cv.imshow('gray', img)

# cv.waitKey(0)
