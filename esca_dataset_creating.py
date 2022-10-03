import tensorflow as tf
 
from tensorflow import keras
 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
 
from tensorflow.keras.preprocessing import image_dataset_from_directory
 
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# Directory

PATH_DATASET_IMAGE = './augmented_esca_dataset_splited'
 
train_data_dir_img = os.path.join(PATH_DATASET_IMAGE, 'train')
validation_data_dir_img = os.path.join(PATH_DATASET_IMAGE, 'validation')
test_data_dir_img = os.path.join(PATH_DATASET_IMAGE, 'test')

PATH_DATASET = './dataset'
 
train_data_dir = os.path.join(PATH_DATASET, 'train_dataset.npy')
validation_data_dir = os.path.join(PATH_DATASET, 'validation_dataset.npy')
test_data_dir = os.path.join(PATH_DATASET, 'test_dataset.npy')

# Parameters 

batch_size = 32
 
nb_train_samples = 14868
nb_validation_samples = 3717
nb_test_samples = 6195
 
n_class = 2
 
epochs = 50

# ***********************************************************************
# **********************        MODEL       *****************************
# ***********************************************************************

start = time.time()

# image size (Model Medium)
img_width, img_height = 320, 180

# input shape
if keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



# ***********************************************************************
# *******************        DATASET       ******************************
# ***********************************************************************

train_dataset = image_dataset_from_directory(train_data_dir_img,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             image_size=(img_width, img_height),
                                             label_mode='categorical')


validation_dataset = image_dataset_from_directory(validation_data_dir_img,
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  image_size=(img_width, img_height),
                                                  label_mode='categorical')


test_dataset = image_dataset_from_directory(test_data_dir_img,
                                            shuffle=True,
                                            batch_size=batch_size,
                                            image_size=(img_width, img_height),
                                            label_mode='categorical')


# preprocessing: input scaling (./255)
train_dataset = train_dataset.map(lambda images, labels: (images/255, labels))
validation_dataset = validation_dataset.map(lambda images, labels: (images/255, labels))
test_dataset = test_dataset.map(lambda images, labels: (images/255, labels))

# Configure the dataset for performance

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# print(type(np.asarray(list(validation_dataset))), np.asarray(list(validation_dataset)).shape)


train_dataset = np.asarray(list(train_dataset))
validation_dataset = np.asarray(list(validation_dataset))
test_dataset = np.asarray(list(test_dataset))

with open('train_dataset.npy', 'wb') as f:
    np.save(f, train_dataset)
with open('validation_dataset.npy', 'wb') as f:
    np.save(f, validation_dataset)
with open(test_data_dir, 'wb') as f:
    np.save(f, test_dataset)
