import sys
import os
import pickle
import pandas as pd

##################  VARIABLES  ##################
BACK_ROOT_DIRECTORY = os.environ.get("BACK_ROOT_DIRECTORY")
sys.path.append(BACK_ROOT_DIRECTORY)
from Lethe.params import *
from Lethe.preprocess.preprocess import preprocess
from Lethe.model.models import XGBClassifier, CNN, LogRegression
from Lethe.model.frequency_modeling import fft_model, fft_prep
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

saved_model  = BACK_ROOT_DIRECTORY + "/Lethe/pickle/model.pkl"
cnn_saved_model  = BACK_ROOT_DIRECTORY + "/Lethe/pickle/model.h5"
saved_pipln  = BACK_ROOT_DIRECTORY + '/Lethe/pickle/pipeline.pkl'
def read_train(smodel):
    
    print("Reading data ... ")
    basedir = BACK_ROOT_DIRECTORY + "/Lethe/raw_data"
    if (smodel == 'fft'):
        X_train, y_train, X_test, y_test, X_val, y_val = fft_prep(basedir, saved_pipln)
    else:
        X_train, y_train, X_test, y_test, X_val, y_val = preprocess(basedir, saved_pipln)
    
    print("Selecting model ... ")
    if smodel == "xgb":
        model = XGBClassifier(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "cnn":
        model = CNN(X_train,y_train, X_test, y_test, X_val, y_val)
        model.save(cnn_saved_model, save_format='tf')
        return model
    elif smodel == "log":
        model = LogRegression(X_train,y_train, X_test, y_test, X_val, y_val) 
    elif smodel == "knn":
        model = KNN(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "svc":
        model = SVC(X_train,y_train, X_test, y_test, X_val, y_val)
    elif smodel == "fft":
        model = fft_model(X_train,y_train, X_test, y_test, X_val, y_val)
        model.save(cnn_saved_model, save_format='tf')
        return model
    
    print ("Model = ", model)

    with open(saved_model, 'wb') as file:
        pickle.dump(model, file)
    return model

def predict(data):

    with open(saved_pipln, 'rb') as file:
        pipeline = pickle.load(file)
    X_fft = pipeline.transform(data.drop(['Sleep_stages'], axis=1))
    y = data['Sleep_stages']

    X = np.hstack((X_fft, y.values.reshape(-1, 1)))

    # CNN
    #if not os.path.exists(cnn_saved_model):
    #    read_train('cnn')
    model = tf.keras.models.load_model(cnn_saved_model)
    return model.predict(X)

    # Not CNN
    if not os.path.exists(saved_model):
        read_train()
    with open(saved_model, 'rb') as file:
        model = pickle.load(file)
    return model.predict(data)

if len(sys.argv) > 1:
    m = sys.argv[1]
    if m not in ['cnn', 'log', 'svc', 'fft', 'xgb', 'knn']:
        print(f'Bad model {m}. Available options are:')
        print('cnn log svc fft xgb knn')
        exit
    else:
        read_train(sys.argv[1])
