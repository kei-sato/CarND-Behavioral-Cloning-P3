import os
import argparse
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def fast():
    # super fast, just check if an error happens
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model


def lenet():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def nvidia():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) # crop 50 pixels from top and 20 pixels from bottom
    model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # nomalize values into [-0.5, 0.5]
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# line fields: center,left,right,steering,throttle,brake,speed
def get_images_steerings(line):
    images = []
    steerings = []

    # center, left, right = line[0], line[1], line[2]
    source_paths = [line[i] for i in range(3)]
    # get file name
    source_paths = [x.split('/')[-1] for x in source_paths]
    # add the data directory path
    source_paths = [os.path.join(_datadir, 'IMG', x) for x in source_paths]
    imgs = [cv2.imread(x) for x in source_paths]
    img_center, img_left, img_right = imgs
    images.extend([img_center, img_left, img_right])

    steering_center = float(line[3])
    # correction of the steering angle for the side camera images
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    steerings.extend([steering_center, steering_left, steering_right])

    # add horizontal flipped images of each three images and nagated steering angles
    image_flips = [np.fliplr(x) for x in imgs]
    steering_flips = [-steering_center, -steering_left, -steering_right]
    images.extend(image_flips)
    steerings.extend(steering_flips)

    return images, steerings


# use generator to avoid running out of memory because the size of the data set could be super large
def generator(lines, batch_size=32):
    num_lines = len(lines)
    while 1: # Loop forever so the generator never terminates
        lines = shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            steerings = []
            for line in batch_lines:
                imgs, steers = get_images_steerings(line)
                images.extend(imgs)
                steerings.extend(steers)

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield shuffle(X_train, y_train)


def train():
    # read csv
    lines = []
    with open(os.path.join(_datadir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # split into the train data and the validation data
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # create generator to avoid running out of memory
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Keras model
    model = nvidia()

    model.compile(loss='mse', optimizer='adam')
    # final data set size will be 6 times larger than the number of the samples because of the augmentation above
    # the augmentations are: center, left, right, flip_center, flip_left, flip_right images
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
        validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=5)

    # save the model to test on the simulator with drive.py
    savepath = 'model.h5'
    if os.path.exists(savepath):
        os.remove(savepath)
    model.save(savepath)


_datadir = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'datadir',
        type=str,
        default='data',
        help='Path to the data directory'
    )
    args = parser.parse_args()
    _datadir = args.datadir # use global variable here as it is used at multiple points
    train()
