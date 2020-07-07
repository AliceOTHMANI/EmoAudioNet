import numpy as np
import os
from os.path import isfile

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

import pandas as pd
from PIL import Image
import librosa
from nlpaug.util.visual.wave import VisualWave
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
from scipy.io import wavfile 
import wave

################################################################################################################
"""
This script is used to get MFCC and Spectrograms features for a same audio file
PS: 
	-	We should compute Spectrograms Images for all audio files before runing this.
	-	We should have labels_audio.csv and labels_spectro.csv
"""
################################################################################################################

################ Path Initialisation
DATA_PATH = "DATA_AUDIO_AUG/labels_audio.csv"
DATA_AUD = "DATA_AUDIO_AUG/"

################ Train Audio File Threshold
TAUX_MFCC = 177

################ Features and Labels lists
features = []
targets_arousal = []
targets_valence = []
file_spectro = []

num = 0
cpt = 0
cpt_ = 0

df = pd.read_csv(DATA_PATH)												# Read audio csv (Audio file name, arousal, valence)

for row in df.iterrows():
    try:
        pourcentage = cpt / df.shape[0]*100								# For state's print

        file = row[1][0]												# Audio Wave Filename 
        targ = row[1][1]												# Arousal Labels
        targ1 = row[1][2]												# Valence Labels

        ################ Load wave file with: 0.5s offset from the start, duration of 2.5s and sample rate of 44100Khz
        X, sr = librosa.load(file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sr=np.array(sr)

        ################ Extaract 13 MFCC features and take their average 
        X = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13),axis=0)

        ################ Threshold: Take only audio files that make sense 
        if len(X) >= TAUX_MFCC : 
            cpt_ += 1 

            ################ Loading state
            if cpt_%150==0:
                print(cpt_, " dans features ... ", pourcentage, " %")

            features.append(X[:TAUX_MFCC])             					# MFCC
            targets_arousal.append(targ)           						# AROUSAL
            targets_valence.append(targ1)  								# VALENCE
            file_spectro.append(file)      								# File to find SPECTRO

        cpt += 1
        if (cpt % 1000) == 0:
            print()
            print(cpt, " loaded from ", df.shape[0])
            print()
    except ValueError as e: 
        pass 

################ Conversion
features = np.array(features)
targets_arousal = np.array(targets_arousal)
targets_valence = np.array(targets_valence)    

################ SPECTROGRAMS FILES ACCORDING TO MFCC's FEATURES (Spectrograms of Thresholded Audio Files )
dico_num = dict()
file_spectro = np.array(file_spectro)
features_SPECTRO = []
targets_SPECTRO_arousal = []
targets_SPECTRO_valence = []

for i in range(0,len(file_spectro)) : 
    file = file_spectro[i] 												# Ex: DATA/16/0.wav
    tab = file.split('/')												# [DATA,16,0.wav]
    file_name = tab[2] 													# 0.wav (We could use [-1])
    num = file_name.split('.')											# [0,wav]
    num = num[0] 														# 0
    dico_num[num] = 0													# Flag it to 0
    
################ Path for Spectro labels
DATA_PATH = "DATA_SPECTRO_AUG_2/labels_spectro.csv"
    
df_spectro = pd.read_csv(DATA_PATH)

for row in df_spectro.iterrows():
    file = row[1][0]
    targ = row[1][1]  													# Arousal -> 1 
    targ1 = row[1][2] 													# Valence -> 2
    
    tab = file.split('/')
    file_name = tab[2] 													
    num = file_name.split('.')
    num = num[0] 
    
    ################ Take Spectrograms of Thresholded Audio files
    if num in dico_num :	
        features_SPECTRO.append(np.array(Image.open(file)))				# Add Spectro
        targets_SPECTRO_arousal.append(targ)
        targets_SPECTRO_valence.append(targ1)
        
features_SPECTRO = np.array(features_SPECTRO)
targets_SPECTRO_arousal = np.array(targets_SPECTRO_arousal)
targets_SPECTRO_valence = np.array(targets_SPECTRO_valence)


################ To verify if we get MFCC and Spectrograms features of same audio files
for i in range(0,len(targets_SPECTRO_arousal)) : 
    if targets_SPECTRO_arousal[i] != targets_arousal[i] :				# If features of a same audio file don't have same labels (If we made a mistake)
        print("ligne "+str(i))
        print(targets_SPECTRO_arousal[i],targets_arousal[i])
        
features_MFCC = features
print(features_SPECTRO.shape)
print(features_MFCC.shape)
print(targets_arousal.shape)
print(targets_valence.shape)
print(targets_SPECTRO_arousal.shape)
print(targets_SPECTRO_valence.shape)


################ Save all features and labels
import pickle

with open('MFCC_ReCoLa.pkl', 'wb') as f:  
     pickle.dump([features_MFCC, targets_arousal, targets_valence], f)
        
with open('SPECTRO_ReCoLa.pkl', 'wb') as f:  
     pickle.dump([features_SPECTRO, targets_arousal, targets_valence], f)
        
