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
from keras.models import load_model as ld

#import sklearn.preprocessing
from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from matplotlib import pyplot as plt

from data_full import full_data_df, data_split, knorm
#--------------------------------------------------------------

def get_data(file_name,data_name, df,remove_ctG=False, dir = './models/'):
    _, _, _, _, _,_, x_test, y_test,_ = data_split(df)
    X_test = knorm(x_test)
    if remove_ctG:
        X_test.drop('ctG', axis = 1, inplace = True)
    else:
        X_test['ctG'] = x_test['ctG']
    mod = ld(dir+file_name)
    y_pred = mod.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    mod_auc = auc(fpr, tpr)
    
    return [fpr, tpr, mod_auc, data_name]

def plot(file_name, *datas, zoom = 0):
    '''
    Parameters
    ----------
    file_name : string
        the name you want to file to have.
    *data : array
        [fpr, tpr, auc, name].
    zoom :int from 0 to 2 , optional
        0 will not give you the zoom of the upper left corner.
        1 will give you this zoomed plot and the normal plot.
        2 will only give you the zoomed plot.
        The default is 0.
    
    Returns
    None.

    '''
    if zoom != 2:
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        for d in datas:
            l = d[3]+' (area= {:.3f})'
            if d[3][0] == 'p':
                plt.plot(d[0],d[1], label= l.format(d[2]))
            else:
                plt.plot(d[0],d[1], label= l.format(d[2]), linestyle = 'dotted')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve for ctG=0')
        plt.legend(loc='best')
        plt.savefig('./plots/'+file_name+'.pdf')
    if zoom != 0:
        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        for d in datas:
            l = d[3]+' (area= {:.3f})'
            plt.plot(d[0],d[1], label= l.format(d[2]))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        plt.savefig('./plots/'+file_name+'zoom.pdf')

def auc_plot():#il manque le 1
    x = [0,1,2,3,5,10]

    df = full_data_df()
    df0 = df.loc[df['ctG']==0]
    df1 = df.loc[df['ctG']==1]
    df2 = df.loc[df['ctG']==2]
    df3 = df.loc[df['ctG']==3]
    df5 = df.loc[df['ctG']==5]
    df10 = df.loc[df['ctG']==10]

    s0 = get_data('0_model.h5','0', df0,remove_ctG=True)
    s1 = get_data('1_model.h5','1', df0,remove_ctG=True)
    s2 = get_data('2_model.h5','2', df2,remove_ctG=True)
    s3 = get_data('3_model.h5','3', df3,remove_ctG=True)
    s5 = get_data('5_model.h5','5', df5,remove_ctG=True)
    s10 = get_data('10_model.h5','10', df10,remove_ctG=True)
    specif = [s0[2],s1[2], s2[2], s3[2], s5[2], s10[2]]
    
    param_minus = []
    param_tot = []
    for dfs in [df0,df1,df2,df3,df5,df10]:
        m = get_data('param_model2.h5','0', dfs)
        param_minus.append(m[2])
        t = get_data('param_tot_model2.h5','0', dfs)
        param_tot.append(t[2])
        




    plt.figure(1)
    plt.plot(x,specif,'rx', label= 'non paramétrique')
    plt.plot(x,specif, 'r--')
    plt.plot(x, param_tot, 'gx', label = 'paramétrique totale')
    plt.plot(x, param_tot, 'g-')
    plt.plot(x, param_minus,'bx', label = 'paramétrique sans 2 et 5')
    plt.plot(x, param_minus, 'b-.')
    plt.xlabel('ctG')
    plt.ylabel('AUC')
    plt.title('AUC selon ctG')
    plt.legend(loc='best')
    plt.savefig('./plots/'+'AUCfinal3'+'.pdf')


#auc_plot()
       
df = full_data_df()
df0 = df.loc[df['ctG']==0]
df1 = df.loc[df['ctG']==1]
df2 = df.loc[df['ctG']==2]
df3 = df.loc[df['ctG']==3]
df5 = df.loc[df['ctG']==5]
df10 = df.loc[df['ctG']==10]
'''
s0 = get_data('0_model.h5','np-0', df0,remove_ctG=True)
s1 = get_data('1_model.h5','np-1', df0,remove_ctG=True)
s2 = get_data('2_model.h5','np-2', df2,remove_ctG=True)
s3 = get_data('3_model.h5','np-3', df3,remove_ctG=True)
s5 = get_data('5_model.h5','np-5', df5,remove_ctG=True)
s10 = get_data('10_model.h5','np-10', df10,remove_ctG=True)
p =  get_data('param_model2.h5','paramétrique', df)
pt = get_data('param_tot_model2.h5','paramétrique total', df)
plot('ROC', s0,s1,s2,s3,s5,s10,p,pt)
'''
s0 = get_data('0_model.h5','np-0', df0,remove_ctG=True)
p =  get_data('param_model2.h5','paramétrique', df0)
pt = get_data('param_tot_model2.h5','paramétrique total', df0)
plot('ROC_ctG0', s0, p,pt)
