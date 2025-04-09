#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from params import *

def preprocess (dir):
    files_dict = {
        "healthy" : 0,
        "narco"   : 1,
        "plm"     : 2,
        "sdb"     : 3, 
        "nfle"    : 4,
        "rbd"     : 5
    }

    columns = []
    for c in range(1,1025):
        columns.append("f"+str(c))
    columns.append('phase')

    df = {}
    for key in files_dict:
        filepath = BACK_ROOT_DIRECTORY + "/Lethe/raw_data/bal_" + key + ".csv"
        df[key] = pd.read_csv(filepath, names=columns)
        df[key]['target'] = files_dict[key]
        df[key]['target'] = df[key]['target'].astype(int)
        df[key].dropna()

    #for key in files_dict:
    #    count = df[key].isna().sum().max()
    #    print(key, " nan values = ", count)

    # Scale
    scaler = RobustScaler()

    for key in files_dict:
        phase = df[key]['phase']
        target = df[key]['target']
        a = scaler.fit_transform(df[key].drop(['target','phase'], axis=1))
        df[key] = pd.DataFrame(a, columns = columns[:-1])
        df[key]['target'] = target
        df[key]['phase'] = phase

    # Merge
    all_df = pd.concat([df['healthy'], df['narco'], df['plm'], df['sdb'], df['nfle'], df['rbd']], axis=0)

    # Delete previous datasets
    for key in files_dict:
       del df[key]

    # Visualization
    #target_0 = all_df[all_df['target'] == 0]
    #observations_0 = target_0.iloc[0]

    #len(all_df[all_df['target'] == 0])

    #target_1 = all_df[all_df['target'] == 1]
    #observations_1 = target_1.iloc[0]

    #x = np.linspace(1,1025,1024)
    #plt.plot(x,observations_0[0:-2], label = "Healthy", color='red')
    #plt.plot(x,observations_1[0:-2], label = "Narco", color='blue')
    #plt.legend()

    #all_df.head(20)

    # Create the feature set
    X = all_df.drop('target', axis=1)
    y = all_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # Balance dataset
    under = RandomUnderSampler(random_state=42)
    X_res, y_res = under.fit_resample(X_train, y_train)

    #X_res.shape
    del X_train, y_train

    return [X_res, y_res, X_test, y_test, X_val, y_val]
