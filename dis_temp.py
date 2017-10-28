import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D, Deconvolution2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
#from IPython import display
from keras.utils import np_utils
from tqdm import tqdm
import keras.backend as K
import pickle
import sys
import math
import tensorflow as tf
from keras.utils import generic_utils
from keras.utils import np_utils

K.set_image_dim_ordering('th')

#define two optimizer with different learning rates
opt = Adagrad(lr=0.005, epsilon=1e-08)
dopt = Adagrad(lr=0.0005, epsilon=1e-08)
dis_temp_opt = Adagrad(lr=0.001, epsilon=1e-08)
opt_enc_frozen = Adagrad(lr=0.008, epsilon=1e-08)

############################ Load data and preprocessing #######################

# load data

##########  N = 20   #############################
# data for Renormalization
print("start loading data")
file = open('/fs1/users/mstieffenhofer/stacked_gan/simple_sgan/data_N20_Trange3200_0.pickle', 'rb')
#file = open('data_N20_Trange3200_0.pickle', 'rb')
X_train1_N20 = pickle.load(file)
file.close() 
file = open('/fs1/users/mstieffenhofer/stacked_gan/simple_sgan/data_N20_Trange3200_1.pickle', 'rb')
#file = open('data_N20_Trange3200_1.pickle', 'rb')
X_train2_N20 = pickle.load(file)
file.close() 

X_train_N20 = np.concatenate((X_train1_N20, X_train2_N20))

print("loading data successfull")


#X_train = X_train[8*3200:22*3200]

(n_samples_b,dim, num_row_b, num_col_b) = X_train_N20.shape

nb_temps_N20 = int(n_samples_b/3200)
print(X_train_N20.shape)

print(nb_temps_N20)

##########  N = 10   #############################
# data for training the discriminator
print("start loading data")
file = open('/fs1/users/mstieffenhofer/stacked_gan/simple_sgan/data1_N10_Trange3200.pickle', 'rb')
X_train1 = pickle.load(file)
file.close() 
file = open('/fs1/users/mstieffenhofer/stacked_gan/simple_sgan/data2_N10_Trange3200.pickle', 'rb')
X_train2 = pickle.load(file)
file.close() 

X_train_all = np.concatenate((X_train1, X_train2))
print("loading data successfull")

#selected temperature range the discriminator shall be trained on
n = 90
m = 120
nb = m-n


for b in range(n,m):
    if b == n:
        #train data
        X_train = X_train_all[b*3200 : (b+1)*3200 -100]
        X_train_v =  X_train_all[b*3200  + 3100: (b+1)*3200 ]
    else:
        #validation data
        samples = X_train_all[b*3200 : (b+1)*3200 -100]
        X_train = np.concatenate((X_train, samples))
        X_train_v = np.concatenate((X_train_v, X_train_all[b*3200  + 3100 : (b+1)*3200 ]))

(n_samples_b,dim, num_row_b, num_col_b) = X_train.shape


print("Shape of X_train:")
print(X_train.shape)
print(X_train_v.shape)

y_train = np.array(3100*[0])
for n in range(1,nb):
   y_train = np.concatenate((y_train,np.array(3100*[n]))) 


y_train_v = np.array(100*[0])
for n in range(1,nb):
   y_train_v = np.concatenate((y_train_v,np.array(100*[n]))) 

print("ja")
labels = ["T = 0.1", "T = 0.2", "T = 0.3","T = 0.4","T = 0.5","T = 0.6","T = 0.7","T = 0.8","T = 0.9","T = 1.0","T = 1.04","T = 1.1","T = 1.2","T = 1.3","T = 1.4","T = 1.5","T = 1.6","T = 1.7","T = 1.8","T = 1.9","T = 2.0"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
print(y_train.shape)
y_train_v = np_utils.to_categorical(y_train_v)
print(y_train_v.shape)

num_classes = y_train.shape[1]
#num_classes_v = y_train_v.shape[1]

print("number of classes:"+str(num_classes))



################################################################################################################################################################################################################################
##################################################################################################  DECODER  (RENORMALIZATION)   #######################################################################################################
################################################################################################################################################################################################################################



def normalize(layers):
    norm = tf.square(layers)
    norm = tf.reduce_sum(norm, 1, keep_dims=True)
    norm = tf.sqrt(norm) 
    norm = tf.concat([norm,norm, norm], 1)
    layers= tf.div(layers, norm)

    return layers    

#########################################----------------------DECODER---------------------################################################

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val	

#########################################----------------------DISCRIMINATOR - TEMPERATURE---------------------################################################
def prob_to_class_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],1])

def prob_to_class(l):
    l = tf.argmax(l, axis = 1)
    l = tf.cast(l, tf.float32)
    return l

	
#dropout rate
dr_dis_tem = 0.3

#define input tenso
dis_temp_inp = Input(shape=(3,10,10))

H = Conv2D(256,(3,3),  subsample=(2,2), border_mode ='same')(dis_temp_inp)

H = LeakyReLU(0.2)(H)

H = Conv2D(512,(3,3), border_mode ='same')(H)

H = LeakyReLU(0.2)(H)

H = Flatten()(H)

H = Dense(2000)(H)
H = LeakyReLU(0.2)(H)

dis_temp_out = Dense(num_classes, activation='softmax')(H)	
	
dis_temp_N10 = Model(dis_temp_inp, dis_temp_out)
dis_temp_N10.compile(loss='categorical_crossentropy', optimizer=dis_temp_opt, metrics=['accuracy'])
dis_temp_N10.summary()	




#Renormalization

encoder_inp = Input(shape=(3,None,None))
H = AveragePooling2D(pool_size=(2, 2))(encoder_inp)
encoder_out = Lambda(normalize)(H)

encoder = Model(encoder_inp, encoder_out)
encoder.compile(loss='categorical_crossentropy', optimizer=opt)
encoder.summary()



#define input tenso
dis_temp_prob_inp = Input(shape=(3,10,10))
H = dis_temp_N10(dis_temp_prob_inp)
dis_temp_prob_out = Lambda(prob_to_class,output_shape=prob_to_class_out_shape)(H)

dis_temp_prob_N10 = Model(dis_temp_prob_inp, dis_temp_prob_out)
#define learning rule
dis_temp_prob_N10.compile(loss='categorical_crossentropy', optimizer=dis_temp_opt, metrics=['accuracy'])
dis_temp_prob_N10.summary()	


"""
class_weight = {0 : 1.,
    1: 1.,
    2: 1.,
    3: 1.,
    4: 1.,
    5: 1.,
    6: 1.,
    7: 1.,
    8: 1.,
    9: 1.,
    10: 1.,
    11: 1.,
    12: 1.,
    13: 1.,
    14: 1.,
    15: 1.,
    16: 1.,
    17: 1.,
    18: 1.,
    19: 1.,
    20: 0.5,
    21: 0.5,
    22: 0.5,
    23: 0.5,
    24: 0.5,
    25: 0.5,
    26: 0.5,
    27: 0.5,
    28: 0.5,
    29: 0.5,
    30: 0.5}
"""


losses = {"dt":[]}  

def training(nb_epoch=5000, BATCH_SIZE=32):
    #display the progess of the learning process    
    #progbar = generic_utils.Progbar(nb_epoch*BATCH_SIZE)
    
            
    make_trainable(dis_temp_N10,True) 

    dt_loss  = dis_temp_N10.fit(X_train, y_train,validation_data=(X_train_v, y_train_v), epochs=4, batch_size=32)
    losses["dt"].append(dt_loss)

    #prog_list = [("DT", dt_loss[0]), ("Acc", dt_loss[1])] 
    #progbar.add(BATCH_SIZE, values= prog_list) 

    dis_temp_N10.save('dis_temp_N10_flow.h5', overwrite=True)
	



for n in range(0,20):
    training(nb_epoch=1000, BATCH_SIZE=32)	
    


