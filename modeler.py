import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
import random as rd

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

import sklearn.preprocessing
from sklearn.model_selection import  train_test_split
from matplotlib import pyplot as plt

from data_full import full_data_df, data_split, knorm
#wtf normalisation ctG
def model(param):#True or False
    if param:
        shape = (22, )
    else: 
        shape = (21, )
    mod = Sequential([
        Dense(units = 60, activation = 'relu', input_shape= shape),
        Dropout(0.5),
        Dense(units = 30, activation = 'relu'),
        Dropout(0.5),
        Dense(units = 15, activation = 'relu'),
        Dropout(0.5),
        Dense(units = 5, activation = 'relu'),
        Dense(units = 1, activation = 'sigmoid')
])
    mod.compile(loss= 'binary_crossentropy', optimizer='adam')
    return mod

print(45*'-=')
df = full_data_df()
ctG_param = 42 #0, 1, 2, 3, 5, 10  pour choisir ctG, 42 pour le paramétrique

if ctG_param != 42:
     param = False
     df = df.loc[df['ctG'] == ctG_param]
     
else : 
    param = True
    ctG_param = 'param'
    print(df['ctG'].unique())
    df = df[df['ctG'] != 2]
    df = df[df['ctG'] != 5]
    
x_train, y_train, w_train, x_val, y_val, w_val, x_test, y_test, w_test = data_split(df)
X_train = knorm(x_train)
X_val = knorm(x_val)
X_test = knorm(x_test)
if param:
    X_train['ctG']= x_train['ctG']
    X_val['ctG'] = x_val['ctG']
else: 
    X_train.drop('ctG', axis=1, inplace=True)
    X_val.drop('ctG', axis = 1, inplace = True)


save_path = './models/'+str(ctG_param)+'_model2.h5'

from os.path import exists

if exists(save_path):
    print("[ERROR] fichier déjà existant")
else:
    mod = model(param)
    es = EarlyStopping(monitor = 'val_loss', patience = 18, verbose = 1)
    mc =  ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True)
    mod.fit(x=X_train, y = y_train, validation_data=(X_val, y_val, np.array(w_val)), batch_size = 125, epochs=200, verbose = 2,sample_weight=np.array(w_train), callbacks = [es, mc])

