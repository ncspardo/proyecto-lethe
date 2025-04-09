import sys
import os
import pickle
import pandas as pd

##################  VARIABLES  ##################
BACK_ROOT_DIRECTORY = os.environ.get("BACK_ROOT_DIRECTORY")
sys.path.append(BACK_ROOT_DIRECTORY)
from Lethe.params import *
from Lethe.preprocess.preprocess import preprocess
from Lethe.model.models import model_selection
from Lethe.model.CNN_model import CNNmodel
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

saved_model  = BACK_ROOT_DIRECTORY + "/Lethe/pickle/model.h5"
def read_train():
    
    print("Reading data ... ")
    basedir = BACK_ROOT_DIRECTORY + "/Lethe/raw_data"
    X_train, y_train, X_test, y_test, X_val, y_val = preprocess(basedir)

    print("Selecting model ... ")
    best_model = model_selection(X_train,y_train, X_test, y_test, X_val, y_val)
    print ("Selected model: " , best_model)

    best_model.save(saved_model, save_format='tf')
    #with open(saved_model, 'wb') as file:
        #pickle.dump(best_model, file)
    return best_model

def predict(input):
    if not os.path.exists(saved_model):
        read_train()
    #with open(saved_model, 'rb') as file:
    #    loaded_model = pickle.load(file)
    model = tf.keras.models.load_model(saved_model)
    return model.predict(input)

