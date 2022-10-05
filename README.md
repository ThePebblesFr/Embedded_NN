# Embedded NN Project

|               |                                 |
|---------------|---------------------------------|
|Authors        |Mickaël JALES, Pierre GARREAU    |
|Status         |Under development                |
|Description    |Embedded Neural Network project dealing with grapevine leaves dataset for early detection and classification of esca disease in vineyards. This code is meant to be executed on STM32L439 board |
|Project        |ISMIN 3A - Embedded IA           |

# Table of contents


## PREPROCESSING

set_samples:  ['train', 'validation', 'test'] 

class:  ['esca' 'healthy']

number of images for class:  [12432 12348] 

split of dataset:
  [[7459 1865 3108]
 [7409 1852 3087]]

## MODEL SMALL

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 80, 45, 32)        896

 activation (Activation)     (None, 80, 45, 32)        0

 max_pooling2d (MaxPooling2D  (None, 40, 22, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 40, 22, 32)        9248

 activation_1 (Activation)   (None, 40, 22, 32)        0

 max_pooling2d_1 (MaxPooling  (None, 20, 11, 32)       0
 2D)

 conv2d_2 (Conv2D)           (None, 20, 11, 64)        18496

 activation_2 (Activation)   (None, 20, 11, 64)        0

 max_pooling2d_2 (MaxPooling  (None, 10, 5, 64)        0
 2D)

 conv2d_3 (Conv2D)           (None, 10, 5, 64)         36928

 activation_3 (Activation)   (None, 10, 5, 64)         0

 max_pooling2d_3 (MaxPooling  (None, 5, 2, 64)         0
 2D)

 conv2d_4 (Conv2D)           (None, 5, 2, 32)          18464

 activation_4 (Activation)   (None, 5, 2, 32)          0

 max_pooling2d_4 (MaxPooling  (None, 2, 1, 32)         0
 2D)

 flatten (Flatten)           (None, 64)                0

 dense (Dense)               (None, 64)                4160

 activation_5 (Activation)   (None, 64)                0

 dropout (Dropout)           (None, 64)                0

 dense_1 (Dense)             (None, 2)                 130

 activation_6 (Activation)   (None, 2)                 0

=================================================================
Total params: 88,322
Trainable params: 88,322
Non-trainable params: 0
_________________________________________________________________

size of images:  80 45
test_result:  [0.5534958243370056, 0.9627118706703186]

## MODEL MEDIUM


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 320, 180, 32)      896

 activation (Activation)     (None, 320, 180, 32)      0

 max_pooling2d (MaxPooling2D  (None, 160, 90, 32)      0
 )

 conv2d_1 (Conv2D)           (None, 160, 90, 32)       9248

 activation_1 (Activation)   (None, 160, 90, 32)       0

 max_pooling2d_1 (MaxPooling  (None, 80, 45, 32)       0
 2D)

 conv2d_2 (Conv2D)           (None, 80, 45, 64)        18496

 activation_2 (Activation)   (None, 80, 45, 64)        0

 max_pooling2d_2 (MaxPooling  (None, 40, 22, 64)       0
 2D)

 conv2d_3 (Conv2D)           (None, 40, 22, 64)        36928

 activation_3 (Activation)   (None, 40, 22, 64)        0

 max_pooling2d_3 (MaxPooling  (None, 20, 11, 64)       0
 2D)

 conv2d_4 (Conv2D)           (None, 20, 11, 32)        18464

 activation_4 (Activation)   (None, 20, 11, 32)        0

 max_pooling2d_4 (MaxPooling  (None, 10, 5, 32)        0
 2D)

 flatten (Flatten)           (None, 1600)              0

 dense (Dense)               (None, 64)                102464

 activation_5 (Activation)   (None, 64)                0

 dropout (Dropout)           (None, 64)                0

 dense_1 (Dense)             (None, 2)                 130

 activation_6 (Activation)   (None, 2)                 0

=================================================================
Total params: 186,626
Trainable params: 186,626
Non-trainable params: 0
_________________________________________________________________

size of images:  320 180
test_result:  [0.1151105985045433, 0.9856335520744324]


# MODEL LARGE 

Not available 


# Commands to execute

* We advise you to create a virtual environment to work on the project. To do this, create an environment at the root of the project with the following commands: 

python -m venv envML *your version of python should be < 3.10, so take care to compile this line with the proper version*

*if your are on Windows*

envML/Scripts/Activate.ps1      <!-- allows to use the virtual python working environment -->

*otherwise*

source envML/bin/activate

deactivate                  <!-- disable the environment -->

*if you have a policy issue with powershell or windows*

Set-ExecutionPolicy -Scope "CurrentUser" -ExecutionPolicy "Unrestricted" <!-- to disable the restrictions -->

Set-ExecutionPolicy -Scope "CurrentUser" -ExecutionPolicy "RemoteSigned" <!-- to enable the restrictions -->


## Install the necessary packages in your own virtual environment (this can also works on your defaut environment)
pip install -r requirements.txt *or* python -m pip install -r requirements.txt

#### in .gitignore, the python virtual environment is ignored
<!-- it allows you to use the python working environment with all necessary packages only on your computer -->
