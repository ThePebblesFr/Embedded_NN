from PIL import Image
import numpy as np
from numpy import random


import os
import pathlib
import random

# directory of dataset
dir_original = "../../data/augmented_esca_dataset"

# name of new dataset
dir_processed = "../../data/augmented_esca_dataset_splited"

# size of new images
size = 1280, 720

#extraction of dataset information
data_dir = pathlib.Path(dir_original)

set_samples = ['train', 'validation', 'test']
print("set_samples: ", set_samples, "\n")

CLASS_NAMES = np.array([item.name for item in sorted(data_dir.glob('*'))])												
print("class: ", CLASS_NAMES, "\n")

N_IMAGES = np.array([len(list(data_dir.glob(item.name+'/*.jpg'))) for item in sorted(data_dir.glob('*'))])			# number of images for class
print("number of images for class: ", N_IMAGES, "\n")

N_samples = np.array([(int(np.around(n*60/100)), int(np.around(n*15/100)), int(np.around(n*25/100))) for n in N_IMAGES])	# number of images for set (train,validation,test)
print("split of dataset: \n ", N_samples, "\n")

# Preprocessing Dataset
# Create the new dataset
# Split Dataset								(also resize and rotate)



# create the dataset folder			***********************************
os.makedirs(dir_processed)

for set_tag in set_samples:
	os.makedirs(dir_processed + '/' + set_tag)

	for class_name in CLASS_NAMES:
		os.makedirs(dir_processed + '/' + set_tag + '/' + class_name)



# SPLIT DATASET (and resize)		*************************************
print("Split dataset.....")

i=0
j=0
k=0
for class_name in CLASS_NAMES:														# "j" cambia con il tipo di pianta [0,3]
	
    print("class name: ", class_name)

    contatore_samples = 0
    k=0

    array = sorted(os.listdir(dir_original + '/' + class_name))
    #random.shuffle(array)

    for image_name in array:	                                       	# "contatore" si azzera ad ogni campo 'train' 'validation' 'test'
	
        print("image: ", i)
        i=i+1

        if contatore_samples==N_samples[j][k]:										    # "k" cambia con train, validation, e test
            k+=1
            contatore_samples=0


        img=Image.open(dir_original +'/'+class_name+'/'+image_name)
        l,_ = img.size
        l=int(l)
        
        
        if l==1080 or l==720:
        
            transposed = img.transpose(Image.ROTATE_90)
            transposed.thumbnail(size)
            transposed.save(dir_processed+'/'+set_samples[k]+'/'+class_name+'/'+image_name)
        
        else:
        
            img.thumbnail(size)
            img.save(dir_processed+'/'+set_samples[k]+'/'+class_name+'/'+image_name)

        contatore_samples+=1	

    j+=1