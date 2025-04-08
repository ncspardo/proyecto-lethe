import sys
import os

##################  VARIABLES  ##################
ROOT_DIRECTORY = os.environ.get("ROOT_DIRECTORY")
sys.path.append(ROOT_DIRECTORY)
from Lethe.preprocess.preprocess import preprocess
from Lethe.model.models import model_selection
from Lethe.model.models import model_selection
from Lethe.params import *

basedir = ROOT_DIRECTORY + "/Lethe/raw_data"
print("Reading data ... ")
X_train, y_train, X_test, y_test = preprocess(basedir)

print("Selecting model ... ")
best_model = model_selection(X_train,y_train, X_test, y_test)
print ("Selected model: " , best_model)
