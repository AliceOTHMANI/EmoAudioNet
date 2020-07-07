import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, Bidirectional, GRU, Concatenate

from keras.layers import Conv1D, MaxPooling1D, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Lambda, Reshape, concatenate
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers


import librosa
import librosa.display
import matplotlib.pyplot as plt

import csv
import cv2
import os
import numpy as np
import random
from random import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

################ Path to save weights
checkpoints_path = "weights_concatenate.h5"

################ Load Features and Targets
with open('MFCC.pkl', 'rb') as f:
    features_MFCC, targets_MFCC = pickle.load(f)
    
with open('SPECTRO.pkl', 'rb') as f:
    features_SPECTRO, targets_SPECTRO = pickle.load(f)

 
NB_ELEM = len(targets_SPECTRO)											# All Elements
TAUX_DATA = 80															# Division Percentage 

train = [i for i in range(0,NB_ELEM)]									# List of elements number to use it like indexes
np.random.shuffle(train)												# Shuffle this list to get features randomly 	

################ Variables initialisation
X_train_Spec2 = []
X_valid_Spec2 = []
Y_train_Spec2 = []
Y_valid_Spec2 = []
X_test_Spec2 = [] 
Y_test_Spec2 = []

X_train_MFCC2= []
Y_train_MFCC2= []
Y_valid_MFCC2= []
X_valid_MFCC2= [] 
X_test_MFCC2= []
Y_test_MFCC2= []

cpt = 0

################ Train/Valid 
for i in train:
    
    if (cpt/NB_ELEM*100) < TAUX_DATA :
        X_train_Spec2.append(features_SPECTRO[i] )
        Y_train_Spec2.append(targets_SPECTRO[i])
        X_train_MFCC2.append(features_MFCC[i])
        Y_train_MFCC2.append(targets_MFCC[i])
    else :
        X_valid_Spec2.append(features_SPECTRO[i])
        Y_valid_Spec2.append(targets_SPECTRO[i])
        X_valid_MFCC2.append(features_MFCC[i])
        Y_valid_MFCC2.append(targets_MFCC[i])
        
    cpt += 1
    

X_train_Spec2 = np.array(X_train_Spec2)
X_valid_Spec2 = np.array(X_valid_Spec2)
Y_train_Spec2 = np.array(Y_train_Spec2)
Y_valid_Spec2 = np.array(Y_valid_Spec2)
X_test_Spec2 = np.array(X_test_Spec2)
Y_test_Spec2 = np.array(Y_test_Spec2)

X_train_MFCC2= np.array(X_train_MFCC2)
Y_train_MFCC2= np.array(Y_train_MFCC2)
Y_valid_MFCC2= np.array(Y_valid_MFCC2)
X_valid_MFCC2= np.array(X_valid_MFCC2)
X_test_MFCC2= np.array(X_test_MFCC2)
Y_test_MFCC2= np.array(Y_test_MFCC2)


from keras.utils import to_categorical

################ ONE HOT ENCODING
Y_train_Spec = to_categorical(Y_train_Spec2, num_classes=10)			# Train Spectrograms
Y_valid_Spec = to_categorical(Y_valid_Spec2, num_classes=10)			# Validation Spectrograms
Y_test_Spec = to_categorical(Y_test_Spec2, num_classes=10)				# Test Spectrograms

Y_train_MFCC = to_categorical(Y_train_MFCC2, num_classes=10)			# Train MFCC 
Y_valid_MFCC = to_categorical(Y_valid_MFCC2, num_classes=10)			# Validation MFCC
Y_test_MFCC = to_categorical(Y_test_MFCC2, num_classes=10)				# Test MFCC

print("***** Split data and One Hot Encoding ... Done *****\n")

################ Shape's print
print("\n                        ----- Spectrogram -----\n\nX_train shape ::",X_train_Spec2.shape)
print("Y_train shape ::",Y_train_Spec.shape)
print("X_valid shape ::",X_valid_Spec2.shape)
print("Y_valid shape ::",Y_valid_Spec.shape)

print("                        ----- MFCC -----\n\nX_train shape ::",X_train_MFCC2.shape)
print("Y_train shape ::",Y_train_MFCC.shape)
print("X_valid shape ::",X_valid_MFCC2.shape)
print("Y_valid shape ::",Y_valid_MFCC.shape)

################ All Elemnts Numbers Confirmation
print("\nNombre Total de données SPECTROGRAM: ", X_train_Spec2.shape[0] + X_valid_Spec2.shape[0]  )
print("Nombre Total de targets SPECTROGRAM: ", Y_train_Spec.shape[0] + Y_valid_Spec.shape[0]  )

print("\nNombre Total de données MFCC: ", X_train_MFCC2.shape[0] + X_valid_MFCC2.shape[0] + X_test_MFCC2.shape[0]  )
print("Nombre Total de targets MFCC: ", Y_train_MFCC.shape[0] + Y_valid_MFCC.shape[0] + Y_test_MFCC.shape[0]  )

import keras.backend as K

################ To Calculate Train Error 
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

################ To Calculate Concordance Correlation Coefficients
def CCC(y_true, y_pred):

    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])			# Normalise
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  		# Normalise

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

    result=top/bottom
    
    print(y_true)


    return K.mean(result)


################################################ Model Creation 

################ Spectrogram Model
ksize = (3,3)
nb_filters = 128
Input_shape = (224,224,3)
Input_shape_2 = (X_train_MFCC2.shape[1], 1)
num_classes = 10
BATCH_SIZE = 32
EPOCH_COUNT = 1000

model1 = Input(shape = Input_shape)
conv_1_Spec = Conv2D(filters = nb_filters, kernel_size = ksize, strides=1, padding= 'same', activation='relu', input_shape = Input_shape, name='Conv_1_Spectrogram')(model1)
conv_2_Spec = Conv2D(filters = nb_filters, kernel_size = ksize, strides=1, padding= 'same', activation='relu', name='Conv_2_Spectrogram')(conv_1_Spec)
droupout = Dropout(0.2)(conv_2_Spec)
maxpool_Spec = MaxPooling2D(pool_size=(8,8))(droupout)
conv_3_Spec = Conv2D(filters = nb_filters, kernel_size = ksize, strides=1, padding= 'same', activation='relu', name='Conv_3_Spectrogram')(maxpool_Spec)
conv_4_Spec = Conv2D(filters = nb_filters, kernel_size = ksize, strides=1, padding= 'same', activation='relu', name='Conv_4_Spectrogram')(conv_3_Spec)
conv_5_Spec = Conv2D(filters = nb_filters, kernel_size = ksize, strides=1, padding= 'same', activation='relu', name='Conv_5_Spectrogram')(conv_4_Spec)
droupout = Dropout(0.2)(conv_5_Spec)
conv_6_Spec = Conv2D(filters = nb_filters, kernel_size = ksize, strides=1, padding= 'same', activation='relu', name='Conv_6_Spectrogram')(droupout)
droupout = Dropout(0.2)(conv_6_Spec)
maxpool_Spec = MaxPooling2D(pool_size=(8,8))(droupout)
flatten = Flatten()(maxpool_Spec)

################ MFCC Model 
model2 = Input(shape = Input_shape_2)
conv_1_MFCC = Conv1D(filters = 128, kernel_size = (5), strides=1, padding= 'same', activation='relu', input_shape = Input_shape_2, name='Conv_1_MFCC')(model2)
conv_2_MFCC = Conv1D(filters = 128, kernel_size = (5), strides=1, padding= 'same', activation='relu', name='Conv_2_MFCC')(conv_1_MFCC)
droupout = Dropout(0.2)(conv_2_MFCC)
maxpool_MFCC = MaxPooling1D(pool_size=(8))(droupout)
flatten1 = Flatten()(maxpool_MFCC)

################ Concatenate CNN MFCC based output and CNN Spectrograms output 
concat = concatenate([flatten, flatten1], axis=-1, name ='concatenation_SPEC/MFCC')


################ Softmax Output
output = Dense(num_classes, activation = 'softmax', name='prediction')(concat)

################################ Create a whole model with: 2 inputs (CNN MFCC based and CNN Spectrograms based) and one output (valence/arousal)
model = Model([model1, model2], output)

################ Optimizers Initialisation
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6)

################ Model Summary
model.summary()

################ Checkpoint
checkpoint = ModelCheckpoint(checkpoints_path, verbose=1, save_weights_only=True, save_best_only=True)

################ Earlystopping
earlyStopping = EarlyStopping(patience = 100, verbose=1)

################ Callbacks
callbacks_list = [checkpoint, earlyStopping]

################ Adapt Input Dimensions for MFCC 
X_train_MFCC = np.expand_dims(X_train_MFCC2, -1)
X_test_MFCC  = np.expand_dims(X_test_MFCC2, -1)
X_valid_MFCC = np.expand_dims(X_valid_MFCC2, -1)

################ Compile Model
model.compile(
        loss = root_mean_squared_error,
        optimizer=opt,
        metrics=['accuracy',CCC]
    )

history = model.fit(x=[X_train_Spec2, X_train_MFCC], y=Y_train_Spec, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, shuffle= False, validation_data=([X_valid_Spec2, X_valid_MFCC], Y_valid_Spec)
, verbose=1, callbacks=callbacks_list)



import matplotlib.pyplot as plt

dic = history.history
print(dic.keys())
plt.plot(dic['val_acc'])
plt.plot(dic['acc'])
plt.figure()
plt.plot(dic['loss'])
plt.plot(dic['val_loss'])
plt.show()
















