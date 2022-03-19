import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

from copy import deepcopy
import random
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Dense, Flatten, Dropout

import sklearn.preprocessing
from sklearn.model_selection import  train_test_split


eft_map_values = {'ctp' : [1,2,3,5,10],
                  'cpt' : [1,2,3,5,10],
                  #'cptb': [1,2,3,5,10],
                  'ctG' : [1,2,3,5,10]}



def drop_and_label(df,  label, col=[]):
    if len(col) > 0:
        df.drop(col, axis=1, inplace=True)
    df["label"] = label
    df["ctp"] = 0
    df["cpt"] = 0
    df["cptb"] = 0
    df["ctG"] = 0
    df["weight"] = 1.

def getEFTdataframe(df, weights, wcoeff='ctp', value=1, signal = True):
    if wcoeff not in eft_map_values:
        print("[ERROR] Wilson Coefficient %s not defined"%wcoeff)
        exit(98)
    else:
        if value not in eft_map_values[wcoeff]:
            print(f"[ERROR] Wilson Coefficient {wcoeff} equal to {value} not defined")
            exit(99)
    if(signal):
        w = weights[f'Hreco__weight_{wcoeff}{value}']
        sig = df[df.label==1]
        sig[wcoeff] = value
        sig['weight'] = w
        return sig
    else:
        bkg = df[df.label==0]
        bkg[wcoeff] = np.random.uniform(1,10, size=data_final_test.label.value_counts()[0])#value
        bkg['weight'] = 1
        return bkg

def getEFTdataframe_WC_bkg(df, weights, wcoeff='ctp', value=1, signal = True):
    if wcoeff not in eft_map_values:
        print(f"[ERROR] Wilson Coefficient {wcoeff} not defined")
        exit(98)
    else:
        if value not in eft_map_values[wcoeff]:
            print(f"[ERROR] Wilson Coefficient {wcoeff} equal to {value} not defined")
            exit(99)

    w = weights[f'Hreco__weight_{wcoeff}{value}']
    sig = df[df.label==0]
    sig[wcoeff] = value
    sig['weight'] = w
    return sig

def full_data_df():

    #EFT weighted samples
    data_ttH_all = uproot.open("TTH_ctcvcp_EFT_Friend.root")['Friends'].arrays(outputtype=pd.DataFrame)
    data_ttW_all = uproot.open("TTW_LO_Friend.root")['Friends'].arrays(outputtype=pd.DataFrame)
    data_ttZ_all = uproot.open("TTZ_LO_Friend.root")['Friends'].arrays(outputtype=pd.DataFrame)


    #less data
    data_ttH = data_ttH_all[:100000]
    #print("len data_ttH",len(data_ttH))
    #data_ttH.head(5)
    data_ttW = data_ttW_all[:100000]
    data_ttZ = data_ttZ_all[:100000]


    columns_drop = data_ttH.columns[data_ttH.columns.str.contains("weight_*")]
    ttH_weights = data_ttH[columns_drop]
    ttW_weights = data_ttW[columns_drop]
    ttZ_weights = data_ttZ[columns_drop]


    drop_and_label(data_ttH, 1, columns_drop)
    drop_and_label(data_ttW, 0, columns_drop)
    drop_and_label(data_ttZ, 0, columns_drop)


    data_final = data_ttH.append([data_ttW,data_ttZ])
    data_final = data_final[data_final.Hreco_dnn_prediction != -99.0]
    data_final = data_final[data_final.Hreco_All5_Jets_pt != -99.0]
    #data_final.head(10)
    #print("len data_final:",len(data_final))
    #print(data_final.head())

    #data_final_test = pd.DataFrame()
    data_final_test=deepcopy(data_final)

    #print(data_final_test.head())

    eft_signals = []

    #data_final.insert(2,'sample','tth' if(data_final[data_final.label]),True)

    for v in eft_map_values['ctG']:
        new_df_s_ctG = getEFTdataframe(data_final_test, ttH_weights, "ctG", v, True)#, 'tth')
        eft_signals.append(new_df_s_ctG)

        new_df_s_ctp = getEFTdataframe(data_final, ttH_weights, "ctp", v, True)#, 'tth')
        eft_signals.append(new_df_s_ctp)

        ttw_ctG = getEFTdataframe_WC_bkg(data_final, ttW_weights, "ctG", v, True)#, 'ttw')
        eft_signals.append(ttw_ctG)

        ttw_ctp = getEFTdataframe_WC_bkg(data_final, ttW_weights, "ctp", v, True)#, 'ttw')
        eft_signals.append(ttw_ctp)

        ttz_ctG = getEFTdataframe_WC_bkg(data_final, ttZ_weights, "ctG", v, True)#, 'ttw')
        eft_signals.append(ttz_ctG)

        ttz_ctp = getEFTdataframe_WC_bkg(data_final, ttZ_weights, "ctp", v, True)#, 'ttw')
        eft_signals.append(ttz_ctp)

        #bkg_df = getEFTdataframe(data_final_test, ttH_weights, "ctG", v, False)#, 'bkg_others')
        #eft_signals.append(bkg_df)

        data_full  = data_final_test.append(eft_signals)
        data_full = data_full[data_full.weight > 0]

    return data_full

def data_split(data_full):#don't normalize

    input_vars_param = ['Hreco_Lep0_pt', 'Hreco_Lep1_pt', 'Hreco_Lep0_eta', 'Hreco_Lep1_eta','Hreco_Lep0_phi', 'Hreco_Lep1_phi', 'Hreco_higgs_reco_mass', 'Hreco_dnn_prediction', 'Hreco_DeltaRl0l1','Hreco_DeltaRClosestJetToLep1', 'Hreco_DeltaPtClosestJetToLep1', 'Hreco_DeltaPtClosestJetToLep0','Hreco_DeltaRClosestJetToLep0','Hreco_HadTop_pt', 'Hreco_HadTop_eta', 'Hreco_HadTop_phi','Hreco_All5_Jets_pt', 'Hreco_All5_Jets_eta','Hreco_met', 'Hreco_met_phi', 'ctG', 'ctp']
    X_train_param, X_test_param, y_train_param, y_test_param, w_train_param, w_test_param = train_test_split(data_full[input_vars_param], data_full.label, data_full.weight, test_size=0.3, random_state=42)
    X_val_param = X_train_param[-20000:]
    y_val_param = y_train_param[-20000:]
    w_val_param = w_train_param[-20000:]
    X_train_param = X_train_param[:-20000]
    y_train_param = y_train_param[:-20000]
    w_train_param = w_train_param[:-20000]
    
    return X_train_param, y_train_param, w_train_param, X_val_param, y_val_param, w_val_param, X_test_param, y_test_param, w_test_param

def knorm(param):
    return keras.utils.normalize(param,axis=-1, order=2)

