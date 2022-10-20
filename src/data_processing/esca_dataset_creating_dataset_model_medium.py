###############################################################################################################################
#                                                                                                                             #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Authors          |Mickaël JALES, Pierre GARREAU      |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Status           |Under development                  |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Description      |This code allows to transform the  |                                                     #
#                                   |photo into a numpy array to improve|                                                     #
#                                   |the computational power during the |                                                     # 
#                                   |training later on.                 |                                                     #
#                                   |This code is for the medium model, |                                                     #
#                                   |photo320x180.                      |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Project          |ISMIN 3A - Embedded IA             |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                                                                                                                             #
###############################################################################################################################


# Only test numpy array for medium model

import numpy as np
from PIL import Image
import os
import time

# Directory

PATH_DATASET_IMAGE = '../../data/augmented_esca_dataset_splited/test'
 
test_esca_data_dir = os.path.join(PATH_DATASET_IMAGE, 'esca')
test_healthy_data_dir = os.path.join(PATH_DATASET_IMAGE, 'healthy')

PATH_DATASET = '../../data/dataset/test'
 
x_test_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_xtest_model_medium.npy')
y_test_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_ytest_model_medium.npy')

# image size (Model Medium)
medium_size = (180, 320)

# Creating xtest and ytest 

first_time = True;

start = time.time()

for img in os.listdir(test_esca_data_dir):
    if first_time:
        image = Image.open(test_esca_data_dir + '/' + img, 'r')
        image = image.resize(medium_size)
        xtest = np.array(image)
        xtest = np.resize(xtest, (1, xtest.shape[0], xtest.shape[1], xtest.shape[2]))
        ytest = np.array([[1., 0.]])
        first_time = False
    else:
        image = Image.open(test_esca_data_dir + '/' + img, 'r')
        image = image.resize(medium_size)
        tmp = np.array(image)
        tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
        xtest = np.concatenate((xtest, tmp), axis=0)
        ytest = np.concatenate((ytest,np.array([[1., 0.]])), axis=0)     # 1 is esca

print("Esca part done")

for img in os.listdir(test_healthy_data_dir):
    image = Image.open(test_healthy_data_dir + '/' + img, 'r')
    image = image.resize(medium_size)
    tmp = np.array(image)
    tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
    xtest = np.concatenate((xtest, tmp), axis=0)
    ytest = np.concatenate((ytest,np.array([[0., 1.]])), axis=0)         # 0 is healthy 

print("Healthy part done")


# Shuffle datasets

xtest_shuffled = np.empty(xtest.shape, dtype=xtest.dtype)
ytest_shuffled = np.empty(ytest.shape, dtype=ytest.dtype)
permutation = np.random.permutation(len(xtest))

for old_index, new_index in enumerate(permutation):
    xtest_shuffled[new_index] = xtest[old_index]
    ytest_shuffled[new_index] = ytest[old_index]

print("Shuffled part done")

# print(Image.fromarray(xtest_shuffled[0]).show())
# print(ytest_shuffled[0])
# print(Image.fromarray(xtest_shuffled[230]).show())
# print(ytest_shuffled[230])


# Save datasets

file_xtest = open(x_test_data_dir, "wb")
file_ytest = open(y_test_data_dir, "wb")

np.save(file_xtest, xtest_shuffled.astype(np.float32))
np.save(file_ytest, ytest_shuffled.astype(np.float32))

file_xtest.close()
file_ytest.close()

print('Time taken for creating dataset of model medium {} sec\n'.format(time.time() - start))
