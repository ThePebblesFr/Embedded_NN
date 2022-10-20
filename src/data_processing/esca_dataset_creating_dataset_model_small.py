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
#                                   |This code is for the small model,  |                                                     #
#                                   |photo80x45.                        |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Project          |ISMIN 3A - Embedded IA             |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                                                                                                                             #
###############################################################################################################################


import numpy as np
from PIL import Image
import os
import time


#################################################################################################################
#                                                                                                               #
#                                               TRAIN                                                           #
#                                                                                                               #
#################################################################################################################

# Directory

PATH_DATASET_IMAGE = '../../data/augmented_esca_dataset_splited/train'
 
train_esca_data_dir = os.path.join(PATH_DATASET_IMAGE, 'esca')
train_healthy_data_dir = os.path.join(PATH_DATASET_IMAGE, 'healthy')

PATH_DATASET = '../../data/dataset/train'
 
x_train_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_xtrain_model_small.npy')
y_train_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_ytrain_model_small.npy')

# image size (Model Small)
small_size = (80, 45)

# Creating xtrain and ytrain 

first_time = True;

start = time.time()

for img in os.listdir(train_esca_data_dir):
    if first_time:
        image = Image.open(train_esca_data_dir + '/' + img, 'r')
        image = image.resize(small_size)
        xtrain = np.array(image)
        xtrain = np.resize(xtrain, (1, xtrain.shape[0], xtrain.shape[1], xtrain.shape[2]))
        ytrain = np.array([[1., 0.]])
        first_time = False
    else:
        image = Image.open(train_esca_data_dir + '/' + img, 'r')
        image = image.resize(small_size)
        tmp = np.array(image)
        tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
        xtrain = np.concatenate((xtrain, tmp), axis=0)
        ytrain = np.concatenate((ytrain,np.array([[1., 0.]])), axis=0)     # "position[0]" is esca

print("Esca part done - train ")

for img in os.listdir(train_healthy_data_dir):
    image = Image.open(train_healthy_data_dir + '/' + img, 'r')
    image = image.resize(small_size)
    tmp = np.array(image)
    tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
    xtrain = np.concatenate((xtrain, tmp), axis=0)
    ytrain = np.concatenate((ytrain,np.array([[0., 1.]])), axis=0)         # "position[1]" is healthy 

print("Healthy part done - train")


# Shuffle datasets

xtrain_shuffled = np.empty(xtrain.shape, dtype=xtrain.dtype)
ytrain_shuffled = np.empty(ytrain.shape, dtype=ytrain.dtype)
permutation = np.random.permutation(len(xtrain))

for old_index, new_index in enumerate(permutation):
    xtrain_shuffled[new_index] = xtrain[old_index]
    ytrain_shuffled[new_index] = ytrain[old_index]

print("Shuffled part done - train")

# print(Image.fromarray(xtrain_shuffled[0]).show())
# print(ytrain_shuffled[0])
# print(Image.fromarray(xtrain_shuffled[230]).show())
# print(ytrain_shuffled[230])


# Save datasets

file_xtrain = open(x_train_data_dir, "wb")
file_ytrain = open(y_train_data_dir, "wb")

np.save(file_xtrain, xtrain_shuffled.astype(np.float32))
np.save(file_ytrain, ytrain_shuffled.astype(np.float32))

file_xtrain.close()
file_ytrain.close()

print('Time taken for creating dataset of model small - train : {} sec\n'.format(time.time() - start))


#################################################################################################################
#                                                                                                               #
#                                               TEST                                                            #
#                                                                                                               #
#################################################################################################################


# Directory

PATH_DATASET_IMAGE = '../../data/augmented_esca_dataset_splited/test'
 
test_esca_data_dir = os.path.join(PATH_DATASET_IMAGE, 'esca')
test_healthy_data_dir = os.path.join(PATH_DATASET_IMAGE, 'healthy')

PATH_DATASET = '../../data/dataset/test'
 
x_test_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_xtest_model_small.npy')
y_test_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_ytest_model_small.npy')

# image size (Model Small)
small_size = (80, 45)

# Creating xtest and ytest 

first_time = True;

start = time.time()

for img in os.listdir(test_esca_data_dir):
    if first_time:
        image = Image.open(test_esca_data_dir + '/' + img, 'r')
        image = image.resize(small_size)
        xtest = np.array(image)
        xtest = np.resize(xtest, (1, xtest.shape[0], xtest.shape[1], xtest.shape[2]))
        ytest = np.array([[1., 0.]])
        first_time = False
    else:
        image = Image.open(test_esca_data_dir + '/' + img, 'r')
        image = image.resize(small_size)
        tmp = np.array(image)
        tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
        xtest = np.concatenate((xtest, tmp), axis=0)
        ytest = np.concatenate((ytest,np.array([[1., 0.]])), axis=0)     # 1 is esca

print("Esca part done - test ")

for img in os.listdir(test_healthy_data_dir):
    image = Image.open(test_healthy_data_dir + '/' + img, 'r')
    image = image.resize(small_size)
    tmp = np.array(image)
    tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
    xtest = np.concatenate((xtest, tmp), axis=0)
    ytest = np.concatenate((ytest,np.array([[0., 1.]])), axis=0)         # 0 is healthy 

print("Healthy part done - test")


# Shuffle datasets

xtest_shuffled = np.empty(xtest.shape, dtype=xtest.dtype)
ytest_shuffled = np.empty(ytest.shape, dtype=ytest.dtype)
permutation = np.random.permutation(len(xtest))

for old_index, new_index in enumerate(permutation):
    xtest_shuffled[new_index] = xtest[old_index]
    ytest_shuffled[new_index] = ytest[old_index]

print("Shuffled part done - test")

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

print('Time taken for creating dataset of model small - test : {} sec\n'.format(time.time() - start))


#################################################################################################################
#                                                                                                               #
#                                               VALIDATION                                                      #
#                                                                                                               #
#################################################################################################################


# Directory

PATH_DATASET_IMAGE = '../../data/augmented_esca_dataset_splited/validation'
 
validation_esca_data_dir = os.path.join(PATH_DATASET_IMAGE, 'esca')
validation_healthy_data_dir = os.path.join(PATH_DATASET_IMAGE, 'healthy')

PATH_DATASET = '../../data/dataset/validation'
 
x_validation_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_xvalidation_model_small.npy')
y_validation_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_yvalidation_model_small.npy')

# image size (Model Small)
small_size = (80, 45)

# Creating xvalidation and yvalidation 

first_time = True;

start = time.time()

for img in os.listdir(validation_esca_data_dir):
    if first_time:
        image = Image.open(validation_esca_data_dir + '/' + img, 'r')
        image = image.resize(small_size)
        xvalidation = np.array(image)
        xvalidation = np.resize(xvalidation, (1, xvalidation.shape[0], xvalidation.shape[1], xvalidation.shape[2]))
        yvalidation = np.array([[1., 0.]])
        first_time = False
    else:
        image = Image.open(validation_esca_data_dir + '/' + img, 'r')
        image = image.resize(small_size)
        tmp = np.array(image)
        tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
        xvalidation = np.concatenate((xvalidation, tmp), axis=0)
        yvalidation = np.concatenate((yvalidation,np.array([[1., 0.]])), axis=0)     # 1 is esca

print("Esca part done - validation ")

for img in os.listdir(validation_healthy_data_dir):
    image = Image.open(validation_healthy_data_dir + '/' + img, 'r')
    image = image.resize(small_size)
    tmp = np.array(image)
    tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
    xvalidation = np.concatenate((xvalidation, tmp), axis=0)
    yvalidation = np.concatenate((yvalidation,np.array([[0., 1.]])), axis=0)         # 0 is healthy 

print("Healthy part done - validation")


# Shuffle datasets

xvalidation_shuffled = np.empty(xvalidation.shape, dtype=xvalidation.dtype)
yvalidation_shuffled = np.empty(yvalidation.shape, dtype=yvalidation.dtype)
permutation = np.random.permutation(len(xvalidation))

for old_index, new_index in enumerate(permutation):
    xvalidation_shuffled[new_index] = xvalidation[old_index]
    yvalidation_shuffled[new_index] = yvalidation[old_index]

print("Shuffled part done - validation")

# print(Image.fromarray(xvalidation_shuffled[0]).show())
# print(yvalidation_shuffled[0])
# print(Image.fromarray(xvalidation_shuffled[230]).show())
# print(yvalidation_shuffled[230])


# Save datasets

file_xvalidation = open(x_validation_data_dir, "wb")
file_yvalidation = open(y_validation_data_dir, "wb")

np.save(file_xvalidation, xvalidation_shuffled.astype(np.float32))
np.save(file_yvalidation, yvalidation_shuffled.astype(np.float32))

file_xvalidation.close()
file_yvalidation.close()

print('Time taken for creating dataset of model small - validation : {} sec\n'.format(time.time() - start))
