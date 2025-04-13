import sys
import os
import pickle
import pandas as pd

##################  VARIABLES  ##################
BACK_ROOT_DIRECTORY = os.getcwd()
from Lethe.params import *
from Lethe.model.frequency_modeling import fft_model, fft_prep
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

saved_model  = BACK_ROOT_DIRECTORY + "/Lethe/pickle/model.h5"
saved_pipln  = BACK_ROOT_DIRECTORY + '/Lethe/pickle/pipeline.pkl'
def read_train():
    
    print("Reading data ... ")
    basedir = BACK_ROOT_DIRECTORY + "/Lethe/raw_data"
    X_train, y_train, X_test, y_test, X_val, y_val = fft_prep(basedir, saved_pipln)
    model = fft_model(X_train,y_train, X_test, y_test, X_val, y_val)
    model.save(saved_model, save_format='tf')
    print ("Model = ", model)
    return model
    
def mpredict(data):

    if not os.path.exists(saved_pipln):
        read_train()
    with open(saved_pipln, 'rb') as file:
        pipeline = pickle.load(file)
    X_fft = pipeline.transform(data.drop(['Sleep_stages'], axis=1))
    y = data['Sleep_stages']

    X = np.hstack((X_fft, y.values.reshape(-1, 1)))

    # CNN
    model = tf.keras.models.load_model(saved_model)
    return model.predict(X)

#read_train()
