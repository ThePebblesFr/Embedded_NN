import numpy as np
from PIL import Image
import os
import time

# Directory

PATH_DATASET_IMAGE = './augmented_esca_dataset_splited/test'
 
test_esca_data_dir = os.path.join(PATH_DATASET_IMAGE, 'esca')
test_healthy_data_dir = os.path.join(PATH_DATASET_IMAGE, 'healthy')

print(test_esca_data_dir)

PATH_DATASET = './dataset'
 
x_test_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_xtest_model_medium.npy')
y_test_data_dir = os.path.join(PATH_DATASET, 'esca_dataset_ytest_model_medium.npy')

# image size (Model Medium)
medium_size = (320, 180)

first_time = True;

start = time.time()

for img in os.listdir(test_esca_data_dir):
    if first_time:
        image = Image.open(test_esca_data_dir + '/' + img, 'r')
        image = image.resize(medium_size)
        xtest = np.array(image)
        xtest = np.resize(xtest, (1, xtest.shape[0], xtest.shape[1], xtest.shape[2]))
        ytest = np.array([["esca"]])
        first_time = False
    else:
        image = Image.open(test_esca_data_dir + '/' + img, 'r')
        image = image.resize(medium_size)
        tmp = np.array(image)
        tmp = np.resize(tmp, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
        xtest = np.concatenate((xtest, tmp), axis=0)
        ytest = np.concatenate((ytest,np.array([["esca"]])), axis=0)

print(xtest.shape)
print(ytest.shape)
print('Time taken for creating dataset of model medium {} sec\n'.format(time.time() - start))