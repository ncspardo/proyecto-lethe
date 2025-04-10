#!/usr/bin/env python
# coding: utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import layers, optimizers, callbacks
import numpy as np
from sklearn.model_selection import GridSearchCV


import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

def XGBClassifier(X_train, y_train, X_test, y_test, X_val, y_val):
    param_grid = {
        'objective': ['multi:softmax', 'multi:softprob'],
        'eval_metric': ['mlogloss', 'merror']
    }
    
    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(
        tree_method='hist',
        max_depth = 4,
        learning_rate=0.1,
        n_jobs=-1,
        num_class=6,                # number of classes in the target
        seed=42
    )
    
    grid_search = GridSearchCV (estimator = model, param_grid = param_grid,
                scoring='accuracy', cv = 5, n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print('Calculating scores ... ')

    # Accuracy score
    print("XGBClassifier accuracy:", accuracy_score(y_test, y_pred))

    # Precision score
    correct_detection_ratio = precision_score(y_test,y_pred, average= 'weighted')
    print("XGBClassifier precision: ", correct_detection_ratio)

    # Recall score
    flag_ratio = recall_score(y_test,y_pred,average = 'weighted')
    print("XGBClassifier recall: ", flag_ratio)

    # F1-score
    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("XGBClassifier f1: ", f1)

    return best_model

def LogisticRegression(X_train, y_train, X_test, y_test, X_val, y_val):
    # Logistic Regression

    # Instanciate model
    print('Trying a LogisticRegression model ... ')
    model = LogisticRegression(max_iter=100, solver='newton-cg')

    model.fit(X_train, y_train)

    #  Validate
    print('Cross validating model .. ')
    #cv_results = cross_validate(model, X_train, y_train, cv=5);
    #print("LogisticRegresion score: ", cv_results['test_score'].mean())

    y_pred = model.predict(X_test)

    print('Calculating scores ... ')
    # Accuracy score
    correct_pred_ratio = accuracy_score(y_test, y_pred)
    print("LogisticRegresion accuracy: ", correct_pred_ratio)

    # Precision score
    correct_detection_ratio = precision_score(y_test,y_pred, average= 'weighted')
    print("LogisticRegresion precision: ", correct_detection_ratio)

    # Recall score
    flag_ratio = recall_score(y_test,y_pred,average = 'weighted')
    print("LogisticRegresion recall: ", flag_ratio)

    # F1-score
    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("LogisticRegresion f1: ", f1)
    return model

def KNN(X_train, y_train, X_test, y_test, X_val, y_val):
    # ConfusionMatrix from predictions
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    #plt.show()

    # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("KNN f1: ", f1)
    return knn


# SVC
def SVC(X_train, y_train, X_test, y_test, X_val, y_val):
    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("SVC f1: ", f1)
    return svc


# CNN model
def CNN(X_train, y_train, X_test, y_test, X_val, y_val):
    print('Trying a CNN + LSTM model ... ')
    
    # Define the model
    model = models.Sequential([
    #layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1],1)),  # 1D convolution layer
    #layers.MaxPooling1D(2),
    #layers.Conv1D(128, 3, activation='relu'),
    #layers.MaxPooling1D(2),
    #layers.Flatten(),
    #layers.Dense(64, activation='relu'),
    layers.GRU(64, activation='tanh'),
    layers.Dense(6, activation='softmax')  # Use softmax for multi-class classification
])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss',
                                       patience=1,
                                       verbose=0,
                                       restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopper]
        )

    # Loss and accuracy
    print("accuracy:     ", history.history['accuracy'][-1])
    print("loss:         ", history.history['loss'][-1])
    print("val accuracy: ", history.history['val_accuracy'][-1])
    print("val loss:     ", history.history['val_loss'][-1])

    return model
