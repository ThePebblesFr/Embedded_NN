###############################################################################################################################
#                                                                                                                             #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Authors          |Mickaël JALES, Pierre GARREAU      |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Status           |Under development                  |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Description      |This code allows to send imputs to |                                                     # 
#                                   |the STM32 board where the model is |                                                     # 
#                                   |already installed.                 |                                                     #
#                                   |This code is for the small model,  |                                                     #
#                                   |photo80x45.                        |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                 |Project          |ISMIN 3A - Embedded IA             |                                                     #
#                 |-----------------|-----------------------------------|                                                     #
#                                                                                                                             #
###############################################################################################################################


import sys, os, array, time
import numpy as np
import serial, math, secrets
from random import *
from ast import literal_eval
import struct

from sklearn import datasets
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt


class timer:
    def __init__(self, name=None):
        self.name = name

    def tic(self):
        self.T_start = time.time()

    def toc(self):
        self.T_stop = time.time()
        print("Elapsed time: " + str(self.T_stop - self.T_start) + " s")



def perso_model_prediction(model, input_values, summary=False):
    # Load an already existing model and test its outputs
    to_be_tested_model = tf.keras.models.load_model(model)

    if summary == True:
        to_be_tested_model.summary()
        for layers in to_be_tested_model.layers:
            print(layers.kernel)

    output_pred = to_be_tested_model.predict(input_values)
    return output_pred


class esca_dataset_set:
    def __init__(self, used_model):
        self.used_model = used_model

        self.x_sample = -1
        self.y_sample = -1

        self.received_output = np.zeros((1,2))
        #self.received_categories = np.zeros((3))

    def set_dataset_from_xtest(self, path_xtest, path_ytest):
        self.X_test = np.load(path_xtest).astype(dtype=np.float32)
        self.Y_test = np.load(path_ytest).astype(dtype=np.float32)

    def pick_rand_value_from_xtest(self):
        rand_sample = randint(0, self.X_test.shape[0]-1)
        self.x_sample = self.X_test[rand_sample]
        self.y_sample = self.Y_test[rand_sample]
        tmp = self.y_sample.argmax(axis=0)
        print("Chosen input's corresponding label is "+str(tmp)+" according to y_test")

    def get_prediction(self):
        print(self.x_sample.shape)
        tmp_proba = perso_model_prediction(self.used_model, self.x_sample)
        self.y_proba = tmp_proba
        tmp = self.y_proba.argmax(axis=0)
        print("\nPYTHON:")
        print("Model prediction is "+str(tmp)+" with probability "+str(self.y_proba[tmp]))
        
    def categorize_received_output(self):
        print ("STM32:")
        print ("Obtained probabilities: \n" + str(self.received_output.round(decimals=4)))
        index = self.received_output.argmax(axis=1)
        print ("Model prediction is "+str(index)+" with probability "+str(self.received_output[0][index])+"\n")

        if (self.y_sample.argmax(axis=0) != index):
            print ("***** Prediction does not match with y_test label")
        else:
            print ("***** Prediction matches with y_test label")

    def match_pred_label(self):
        if (self.y_sample.argmax(axis=0) != self.received_output.argmax(axis=1)):
            return 1
        else:
            return 0



def synchronisation_with_target(debug=False):
    sync = False
    ret = None

    while (sync == False):
        ser.write(b"sync")
        ret = ser.read(3)
        if (ret == b"110"): # "101" has been chosen arbitrarily
            sync = True
            if (debug):
                print("Synchronised")
        else:
            if (debug):
                print ("Wrong ack reponse")


# image size (Model Small)
small_size = (45, 80, 3)

def send_NN_inputs_to_STM32(esca_dataset_set, ser):
    if not ser.isOpen():
        print ("Error: serial connection to be used isn't opened")
        sys.exit(-1)

    # Synchronisation loop
    synchronisation_with_target()

    # Send inputs to the Neural Network
    input_sent = False
    ser.flush()

    tmp = esca_dataset_set.x_sample
    while(input_sent == False):

        for i in range(small_size[0]):
            for j in range(small_size[1]):
                for m in range(small_size[2]):
                    ser.write(tmp[i,j,m])

        input_sent = True

    # Used for debug (i.e. get the picture sent)
    #for i in range(28):
    #    for j in range(28):
    #        tmp[i][j] = struct.unpack('f', ser.read(4))[0]
    #plt.imshow(tmp, cmap='gray')
    #plt.show()

    # wait for the output values generated by the STM32
    out_ack = b"000"
    while(out_ack != b"110"): # "010" has been chosen arbitrarily
        out_ack = ser.read(3)

    for i in range(0, 2):
        predi_values = ser.read(4)
        esca_dataset_set.received_output[0][i] = struct.unpack('f', predi_values)[0]

    esca_dataset_set.categorize_received_output()



if __name__ == '__main__':

    tf.autograph.set_verbosity(0)
    nb_inference = 1
    with serial.Serial("COM10", 115200, timeout=1) as ser:      # ! You need to put your port COM corresponding
        chrono = timer("Chrono")
    
        # Model available for board's results comparaison
        used_model = "data/h5/model_small_b32.h5"
    
        # X_test and Y_test dataset available for inference
        path_xtest = "data/dataset/test/esca_dataset_xtest_model_small2.npy"
        path_ytest = "data/dataset/test/esca_dataset_ytest_model_small2.npy"
        
    
        i = 0
        nb_error = 0
        errored_elem = []
    
        chrono.tic()
        while(i < nb_inference):
            print ("\n\n----------- Inference "+str(i)+" requested: -----------\n")
    
            t1 = esca_dataset_set(used_model)
            t1.set_dataset_from_xtest(path_xtest, path_ytest)
            t1.pick_rand_value_from_xtest()
    
            print ("\n")
    
            send_NN_inputs_to_STM32(t1, ser)
    
            if(t1.match_pred_label() == 1):
                nb_error += 1
                errored_elem.append(t1)
    
            i = i + 1
    
            del t1
    
        print ('\n')
        chrono.toc()
        print ("\nAll inferences have been effected")
        print ("\nNumber of error reported according to y_test: "+str(nb_error))
        ser.close()
