import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

temp = []
for i in range(1, 657):
    if i < 100:
        if i < 10:
            temp.append('img00' + str(i))
        else:
            temp.append('img0' + str(i))
    else:
        temp.append('img' + str(i))

train_path = '../Temp/Img'

train_batches = ImageDataGenerator() \
    .flow_from_directory(directory=train_path, target_size=(300, 400), classes=temp, batch_size=10)


# df = pd.read_csv(
#     "../Kannada_Images/kannada.csv", dtype={"img": pd.StringDtype()}
# )
# df["class"] = df["class"].astype("category")
# values = list(df.columns.values)
# label_array = df[values[-1:]]
# label_array = np.array(label_array, dtype='int')
# np.save('./label_array', label_array)
label_array = np.load('./label_array.npy')
print(label_array[35])
print(label_array.shape)

# img_array = []
# files = glob.glob('../Temp/Img/*.png')
# for file in files:
#     img = cv.imread(file, cv.IMREAD_GRAYSCALE)
#     img = cv.resize(img, (400, 300))
#     # cv.imshow('Temp', img)
#     # cv.waitKey(0)
#     img_array.append(img)

# img_array = np.array(img_array)
# np.save('./img_array', img_array)
img_array = np.load('./img_array.npy')
img_array = img_array.reshape(
    img_array.shape[0], img_array.shape[1], img_array.shape[2], 1)
print(img_array[35])
print(img_array.shape)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
           padding='same', input_shape=(300, 400, 1)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=img_array, y=label_array, batch_size=10, epochs=2, verbose=2)
