#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from params import *
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle

def preprocess (dir, pfile):
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

    newcols = []
    for c in range(1,3073):
        newcols.append("f"+str(c))

    dict = {}
    for key in files_dict:
        filepath = BACK_ROOT_DIRECTORY + "/Lethe/raw_data/bal_" + key + ".csv"
        df = pd.read_csv(filepath, names = columns)
        df1 = df[df['phase'] == 1].drop(['phase'], axis=1)
        df2 = df[df['phase'] == 2].drop(['phase'], axis=1)
        df3 = df[df['phase'] == 3].drop(['phase'], axis=1)
    
        combined = np.concatenate([df1.values, df2.values, df3.values], axis=1)

        newdf = pd.DataFrame(combined, columns=newcols)

    #    filename = '/bigd/code/grinbea/lethe-website/test_data/'+key+'123.csv'
    #    newdf.to_csv(filename,index=False)

        newdf['target'] = files_dict[key]
        dict[key] = pd.DataFrame(newdf)
        del df, df1, df2, df3

    # Merge
    all_df = pd.concat([dict['healthy'], dict['narco'], \
            dict['plm'], dict['sdb'], dict['nfle'], dict['rbd']], axis=0)
    all_df.dropna()

    # Delete previous datasets
    for key in files_dict:
       del dict[key]

    y = all_df['target']

    # Balance dataset
    #under = RandomUnderSampler(random_state=42)
    #X_res, y_res = under.fit_resample(X_train, y_train)
    #del X_train, y_train

    # Encode 'phase'
    #X = encode(all_df)

    # Pipeline
    pipeline = Pipeline ([
#        ('pca', PCA(n_components=500)),
        ('scaler', MinMaxScaler())
    ])

    X = pipeline.fit_transform(all_df.drop(['target'], axis=1))

    # Save the pipeline
    with open(pfile, 'wb') as file:
        pickle.dump(pipeline, file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    return [X_train, y_train, X_test, y_test, X_val, y_val]

def encoder (all_df):
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(X['phase'].values.reshape(-1, 1))
    enc_data = encoded_data.toarray()
    
    # Create the feature set
    X = all_df.drop(['target','phase'], axis=1)
    X['phase_1'] = enc_data[:,0]
    X['phase_2'] = enc_data[:,1]
    X['phase_3'] = enc_data[:,2]
    X['phase_4'] = enc_data[:,3]
    return X
