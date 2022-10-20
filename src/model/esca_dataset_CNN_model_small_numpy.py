###############################################################################################################################
#                                                                                                                             #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Authors          |Mickaël JALES, Pierre GARREAU      |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Status           |Under development                  |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Description      |This code train the model from the |                                                     # 
#                                   |pre-processed numpy arrays.        |                                                     #
#                                   |This code is for the small model,  |                                                     #
#                                   |photo80x45.                        |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Project          |ISMIN 3A - Embedded IA             |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                                                                                                                             #
###############################################################################################################################


import sys, os, array, time
import numpy as np
import matplotlib.pyplot as plt
import IPython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class timer:
    def __init__(self, name=None):
        self.name = name
        self.T_start = -1
        self.T_stop  = -1

    def tic(self):
        self.T_start = time.time()

    def toc(self):
        self.T_stop = time.time()

    def res(self):
        if (self.T_start == -1) or (self.T_stop == -1):
            print("Error: Measurement cannot be done")
        else:
            return str(self.T_stop - self.T_start)



def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.title('Training and Validation Loss_80x45')
    plt.savefig("../../data/h5/loss_model_small.png")
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')    
    plt.title('Training and Validation Accuracy_80x45')
    plt.savefig("../../data/h5/accuracy_model_small.png")
    plt.show()

def load_mnist_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

def build_model(data):

        model_small = Sequential()
        model_small.add(Conv2D(32, (3, 3), padding='same', input_shape=data.input_shape))
        model_small.add(Activation('relu'))
        model_small.add(MaxPooling2D(pool_size=(2, 2)))
        
        model_small.add(Conv2D(32, (3, 3), padding='same'))
        model_small.add(Activation('relu'))
        model_small.add(MaxPooling2D(pool_size=(2, 2)))
        
        model_small.add(Conv2D(64, (3, 3), padding='same'))
        model_small.add(Activation('relu'))
        model_small.add(MaxPooling2D(pool_size=(2, 2)))
        
        model_small.add(Conv2D(32, (3, 3), padding='same'))
        model_small.add(Activation('relu'))
        model_small.add(MaxPooling2D(pool_size=(2, 2)))
        
        model_small.add(Flatten())
        model_small.add(Dense(64))
        model_small.add(Activation('relu'))
        model_small.add(Dropout(0.5))
        model_small.add(Dense(2))           #because we have 2 class
        model_small.add(Activation('softmax'))
        
        # model_small.summary()
                
        return model_small

        # # Small CNN for MNIST recognition
        # model = models.Sequential()
        
        # # Dense layer
        # model.add(layers.Conv2D(2, (3, 3), padding='same', activation='relu', input_shape=data.input_shape))
        # model.add(layers.MaxPooling2D((2, 2), padding='valid'))
        # model.add(layers.Flatten())
        
        # # Dense layer
        # model.add(layers.Dense(16, activation='relu'))
        
        # # Output layer
        # model.add(layers.Dense(2, activation='softmax'))
                
        # return model
    
class dataset:
    def __init__(self):

        self.x_train        = np.load('../../data/dataset/train/esca_dataset_xtrain_model_small.npy')
        self.y_train        = np.load('../../data/dataset/train/esca_dataset_ytrain_model_small.npy')
        self.x_test         = np.load('../../data/dataset/test/esca_dataset_xtest_model_small.npy')
        self.y_test         = np.load('../../data/dataset/test/esca_dataset_ytest_model_small.npy')
        self.x_validation   = np.load('../../data/dataset/validation/esca_dataset_xvalidation_model_small.npy')
        self.y_validation   = np.load('../../data/dataset/validation/esca_dataset_yvalidation_model_small.npy')
        
        # Rescale of images
        self.x_train        = self.x_train / 255.0
        self.x_test         = self.x_test / 255.0
        self.x_validation   = self.x_validation / 255.0
        
        self.input_shape = self.x_train.shape[1:]
        
        
        # # Transform label to one hot vector
        # self.y_train        = tf.keras.utils.to_categorical(self.y_train, 2)
        # self.y_validation   = tf.keras.utils.to_categorical(self.y_validation, 2)
        # self.y_test         = tf.keras.utils.to_categorical(self.y_test, 2)
        
        self.nb_epochs  = 50
        self.batch_size = 32
        
        print("Number training examples:  ", len(self.x_train))
        print("Number test examples:      ", len(self.x_test))
        print("Number validation examples:", len(self.x_validation))
        print("\n")
        print("\tTrain Dataset      --> x_train:        " + str(np.shape(self.x_train))         + "    y_train:         " + str(np.shape(self.y_train)))
        print("\tValidation Dataset --> x_validation:   " + str(np.shape(self.x_validation))    + "    y_validation:    " + str(np.shape(self.y_validation)))
        print("\tTesting Dataset    --> x_test:         " + str(np.shape(self.x_test))          + "    y_test:          " + str(np.shape(self.y_test)))
        print("\tNumber of epochs:  "+str(self.nb_epochs))
        print("\tBatch size:        "+str(self.batch_size))
        print("\n")
        
    # def MLP_input_data_preparation(self):
    #     self.input_shape    = np.prod(self.x_train.shape[1:])
    #     self.x_train        = self.x_train.reshape(self.x_train.shape[0], self.input_shape)
    #     self.x_validation   = self.x_validation.reshape(self.x_validation.shape[0], self.input_shape)
    #     self.x_test         = self.x_test.reshape(self.x_test.shape[0], self.input_shape)
        
    #     print ("\n")
    #     print ("New dimensions after MLP reshape:\n")
    #     print ("Train Dataset      --> x_train:         " + str(np.shape(self.x_train))         + "    y_train:         " + str(np.shape(self.y_train)))
    #     print ("Validation Dataset --> x_validation:    " + str(np.shape(self.x_validation))    + "    y_validation:    " + str(np.shape(self.y_validation)))
    #     print ("Testing Dataset    --> x_test:          " + str(np.shape(self.x_test))          + "    y_test:          " + str(np.shape(self.y_test)))
    #     print ("\n")
    
        

def train_model(dataset):
    
    print("Preparing context to train of the model ...")
    chronos = timer()

    x_train             = dataset.x_train
    x_validation        = dataset.x_validation
    y_train             = dataset.y_train
    y_validation        = dataset.y_validation
    batch_size          = dataset.batch_size
    epochs              = dataset.nb_epochs
    
    l_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(lr=l_rate)
    
    model = build_model(dataset)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    
    chronos.tic()
    print("### START TRAINING ###")

    es = EarlyStopping(monitor='val_loss', patience=10)
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_validation, y_validation),
                        shuffle=True,
                        callbacks=[es])
    
    chronos.toc()
    print("\n### STOP TRAINING ###")

    train_scores        = model.evaluate(x_train, y_train, verbose=0)
    validation_scores   = model.evaluate(x_validation, y_validation, verbose=0)

    print("\n")
    print('***** Train loss:            ', train_scores[0])
    print('***** Train accuracy:        ', train_scores[1])
    print("")
    print('***** Validation loss:       ', validation_scores[0])
    print('***** Validation accuracy:   ', validation_scores[1])
    print("\n")
    
    return model, history



def test_model(dataset, trained_model, save_pred=False, trust_indicator=False):
    chronos = timer()
    
    x_test = dataset.x_test
    y_test = dataset.y_test
    
    print("### START TESTING ###")
    chronos.tic()
        
    test_score = trained_model.evaluate(x_test, y_test, verbose=0)
    
    chronos.toc()
    print("")
    print('***** Test loss:         ', test_score[0])
    print('***** Test accuracy:     ', test_score[1])
    print("")
    
    if trust_indicator:
        output_pred = trained_model.predict(x_test)
        output_pred.sort(axis=1)
        
        trust_diff = [(output_pred[i][-1] - output_pred[i][-2]) for i in range(x_test.shape[0])]
        trust_diff = np.array(trust_diff)
        
        trust_mean = trust_diff.mean(axis=0)
        trust_min  = np.amin(trust_diff)
        
        print("")
        print("***** Trust difference mean  :", trust_mean)
        print("***** Minimal difference     :", trust_min)
        print("")


        
if __name__ == '__main__':
    
    print("Start process ... \n")
    
    data = dataset()
    
    # Training of Neural Network
    trained_model, history = train_model(data)
    test_model(data, trained_model)
    
    # Plot accuracy and loss 
    plot_history(history)

    # Saving model and testing datasets
    PATH_MODELS = '../../data/h5'
 
    name_model_small = os.path.join(PATH_MODELS, 'model_small_b32.h5')

    trained_model.save(name_model_small)
