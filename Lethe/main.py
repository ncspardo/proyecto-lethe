import sys
import os
import pickle
import pandas as pd

##################  VARIABLES  ##################
BACK_ROOT_DIRECTORY = os.environ.get("BACK_ROOT_DIRECTORY")
sys.path.append(BACK_ROOT_DIRECTORY)
from Lethe.params import *
from Lethe.preprocess.preprocess import preprocess
from Lethe.model.models import XGBClassifier, CNN
from Lethe.model.CNN_model import CNNmodel
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

saved_model  = BACK_ROOT_DIRECTORY + "/Lethe/pickle/model.pkl"
saved_pipln  = BACK_ROOT_DIRECTORY + '/Lethe/pickle/pipeline.pkl'
def read_train(smodel):
    
    print("Reading data ... ")
    basedir = BACK_ROOT_DIRECTORY + "/Lethe/raw_data"
    X_train, y_train, X_test, y_test, X_val, y_val = preprocess(basedir, saved_pipln)

    print("Selecting model ... ")
    if smodel == "xgb":
        model = XGBClassifier(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "cnn":
        model = CNNmodel(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "log":
        model = LogisticRegression(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "knn":
        model = KNN(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "svc":
        model = SVC(X_train,y_train, X_test, y_test, X_val, y_val)
    
    print ("Model = ", model)

    #best_model.save(saved_model, save_format='tf')
    with open(saved_model, 'wb') as file:
        pickle.dump(model, file)
    return model

def predict(input):
    with open(saved_pipln, 'rb') as file:
        pipeline = pickle.load(file)
    
    data = pipeline.transform(input)

    if not os.path.exists(saved_model):
        read_train()
    with open(saved_model, 'rb') as file:
        model = pickle.load(file)
    #model = tf.keras.models.load_model(saved_model)
    return model.predict(data)

#read_train('xgb')
