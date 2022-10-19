## Data augmentation

dataset_name = "esca_dataset"

# Input pameters choosen

transformation_array = [
                        "horizontalFlip",
                        "verticalFlip", 
                        "rotation", 
                        "widthShift", 
                        "heightShift",  
                        "shearRange",
                        "zoom", 
                        "blur",
                        "brightness", 
                        "contrast",
                        "saturation",
                        "hue",
                        "gamma"
                        ];
enable_show = False;

# The new dataset 'augmented_esca_dataset' will be created.
# This dataset contains the augmented images create by the orginal images, in order to obtain an expanded version of the orginal dataset 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
import os
import shutil
from numpy import expand_dims
import cv2
import matplotlib.pyplot as plt
from pathlib import Path



def blur(img):
    return (cv2.blur(img,(30,30)))

def horizontal_flip(img):
    return (tf.image.flip_left_right(img))

def vertical_flip(img):
    return (tf.image.flip_up_down(img))
 
def contrast(img):
    return (tf.image.adjust_contrast(img, 0.5))

def saturation(img):
    return (tf.image.adjust_saturation(img, 3))

def hue(img):
    return (tf.image.adjust_hue(img, 0.1)) 

def gamma(img):
    return (tf.image.adjust_gamma(img, 2))



new_dataset = 'augmented_esca_dataset'
classes = ['esca', 'healthy']
for class_tag in classes:
  input_path = '../../data/' + dataset_name + '/' + class_tag + '/'
  output_path = '../../data/' + new_dataset + '/' + class_tag + '/'
  print(input_path)
  print(output_path)
  # TMP
  #shutil.rmtree(output_path)
  # END TMP
  print(os.path.exists(output_path))
  try:
    if not os.path.exists(output_path):
      os.makedirs(output_path)
  except OSError:
      print ("Creation of the directory %s failed\n\n" % output_path)
  else:
      print ("Successfully created the directory %s\n\n" % output_path)

  for filename in os.listdir(input_path):
    if filename.endswith(".jpg"):
      # Copy the original image in the new dataset
      original_file_path = input_path + filename
      original_newname_file_path = output_path + Path(filename).stem + "_original.jpg"
      shutil.copy(original_file_path, original_newname_file_path)
      # Initialising the ImageDataGenerator class. 
      # We will pass in the augmentation parameters in the constructor. 
      for transformation in transformation_array:
        if transformation == "horizontalFlip":
              #datagen = ImageDataGenerator(horizontal_flip = True)                 # for random flip
              datagen = ImageDataGenerator(preprocessing_function=horizontal_flip)  # all imgs flipped
        elif transformation == "verticalFlip":
              #datagen = ImageDataGenerator(vertical_flip = True)                   # for random flip
              datagen = ImageDataGenerator(preprocessing_function=vertical_flip)    # all imgs flipped
        elif transformation == "rotation":
              datagen = ImageDataGenerator(rotation_range = 40, fill_mode='nearest') 
        elif transformation == "widthShift":
              datagen = ImageDataGenerator(width_shift_range = 0.2, fill_mode='nearest')
        elif transformation == "heightShift":
              datagen = ImageDataGenerator(height_shift_range = 0.2, fill_mode='nearest')         
        elif transformation == "shearRange":
              datagen = ImageDataGenerator(shear_range = 0.2)   
        elif transformation == "zoom":
              datagen = ImageDataGenerator(zoom_range = [0.5, 1.0])
        elif transformation == "blur":
              datagen = ImageDataGenerator(preprocessing_function=blur)        
        elif transformation == "brightness":
              #Values less than 1.0 darken the image, e.g. [0.5, 1.0], 
              #whereas values larger than 1.0 brighten the image, e.g. [1.0, 1.5], 
              #where 1.0 has no effect on brightness.
              datagen = ImageDataGenerator(brightness_range = [1.1, 1.5])
        elif transformation == "contrast": 
              datagen = ImageDataGenerator(preprocessing_function=contrast)
        elif transformation == "saturation": 
              datagen = ImageDataGenerator(preprocessing_function=saturation)      
        elif transformation == "hue": 
              datagen = ImageDataGenerator(preprocessing_function=hue)    
        elif transformation == "gamma": 
              datagen = ImageDataGenerator(preprocessing_function=gamma)      

        # Loading a sample image 
        img = load_img(input_path + filename) 
        # Converting the input sample image to an array 
        data = img_to_array(img) 
        # Reshaping the input image expand dimension to one sample
        samples = expand_dims(data, 0) 
        # Plot original image
        print("Original image:")
        print(filename)
        if enable_show:
          plt.imshow(img)
          plt.show()
          print("\n\n")

        # Generating and saving n_augmented_images augmented samples
        print("Apply " + transformation + ".")
        # prepare iterator
        it = datagen.flow(samples, batch_size = 1, 
                    save_to_dir = output_path, 
                    save_prefix = Path(filename).stem + "_" + transformation,
                    save_format ='jpg')
        batch = it.next()
        # Plot trasnformed image
        image = batch[0].astype('uint8')
        if enable_show:
          print("Transformed image:")
          plt.imshow(image)
          plt.show()
        print("\n\n")

print("Done!\n\n")