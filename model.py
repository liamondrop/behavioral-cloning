import os
import csv
import cv2
import time
import random
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPool2D

DATA_DIR = os.path.abspath('data')
MODEL_DIR = os.path.abspath('model')
LOGS_DIR = os.path.abspath('logs')
LOG_FILE = 'driving_log.csv'
EPOCHS = 5
VALIDATION_SIZE = 0.2
USE_GENERATOR = False
MODEL_NAME = "model_{}".format(int(time.time()))

# Log loss and validation loss per epoch
from keras.callbacks import CSVLogger
csv_logger = CSVLogger(os.path.join(LOGS_DIR, MODEL_NAME) + ".log")

# Define the model
model = Sequential()
model.add(Cropping2D(((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.) - 0.5))
model.add(Conv2D(24, (5,5), strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu"))
model.add(Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Utility functions
def preprocess_img(img):
    """ minimize variance in light and shadow by using histogram equalization
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def get_camera_offset(angle, max_offset=.25, min_offset=0.1):
    """ correct for left and right angled cameras with dynamic offset
        As angle increases, offset also increases, but not above the `max_angle`
        and not below the `min_offset`
    """
    extra_offset = min(angle*(angle/max_offset), max_offset)
    return min_offset + extra_offset

def process_image_data(data):
    images = []
    angles = []
    for row in image_data:
        # center, left, and right images, preprocessed with histogram equalization
        image_c = preprocess_img(cv2.imread(os.path.join(DATA_DIR, row[0].strip())))
        image_l = preprocess_img(cv2.imread(os.path.join(DATA_DIR, row[1].strip())))
        image_r = preprocess_img(cv2.imread(os.path.join(DATA_DIR, row[2].strip())))

        # create adjusted steering measurements for the side camera images
        angle_c = float(row[3])
        offset = get_camera_offset(angle_c)
        angle_l = angle_c + offset
        angle_r = angle_c - offset

        # randomly flip images and angles to correct for any left/right bias
        if random.randint(0,1) == 1:
            image_c = np.fliplr(image_c)
            angle_c = -angle_c
        if random.randint(0,1) == 1:
            image_l = np.fliplr(image_l)
            angle_l = -angle_l
        if random.randint(0,1) == 1:
            image_r = np.fliplr(image_r)
            angle_r = -angle_r

        # add images and angles to data set
        images.extend((image_c, image_l, image_r))
        angles.extend((angle_c, angle_l, angle_r))

    return skl.utils.shuffle(np.array(images), np.array(angles))

def data_generator(data, batch_size=32):
    data_size = len(data)
    while 1:
        skl.utils.shuffle(data)
        for offset in range(0, data_size, batch_size):
            batch = data[offset:offset+batch_size]
            yield process_image_data(batch)

# Load the image_data from the driving log
image_data = []
with open(os.path.join(DATA_DIR, LOG_FILE)) as handle:
    reader = csv.reader(handle)
    for row in reader:
        image_data.append(row)

if USE_GENERATOR:
    # compile and train the model using the generator function
    # (more memory efficient but considerably slower)
    data_train, data_valid = train_test_split(image_data, test_size=VALIDATION_SIZE)
    train_gen = data_generator(data_train)
    valid_gen = data_generator(data_valid)
    model.fit_generator(train_gen,
                        steps_per_epoch=len(data_train) * 3,
                        validation_data=valid_gen,
                        validation_steps=len(data_valid) * 3, epochs=EPOCHS)
else:
    X_train, y_train = process_image_data(image_data)
    model.fit(X_train, y_train,
              validation_split=VALIDATION_SIZE,
              epochs=EPOCHS,
              callbacks=[csv_logger])

model.save(os.path.join(MODEL_DIR, MODEL_NAME) + ".h5")
