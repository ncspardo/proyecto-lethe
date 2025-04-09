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

def model_selection(X_train, y_train, X_test, y_test, X_val, y_val):
    # Logistic Regression

    # Instanciate model
    print('Trying a LogisticRegression model ... ')
    model = LogisticRegression(max_iter=100, solver='newton-cg')

    model.fit(X_train, y_train)

    #  Validate
    print('Cross validating model .. ')
    cv_results = cross_validate(model, X_train, y_train, cv=5);
    print("LogisticRegresion score: ", cv_results['test_score'].mean())

    y_pred = model.predict(X_test)

    print('Calculating scores ... ')
    # Accuracy score
    correct_pred_ratio = accuracy_score(y_test, y_pred)
    print("LogisticRegresion accuracy: ", correct_pred_ratio)
    correct_pred_ratio

    # Precision score
    correct_detection_ratio = precision_score(y_test,y_pred, average= 'weighted')
    print("LogisticRegresion precision: ", correct_detection_ratio)

    # Recall score
    flag_ratio = recall_score(y_test,y_pred,average = 'weighted')
    print("LogisticRegresion recall: ", flag_ratio)

    # F1-score
    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("LogisticRegresion f1: ", f1)

    # ConfusionMatrix from predictions
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    #plt.show()

    # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("KNN f1: ", f1)

    # SVC
    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    f1 = f1_score(y_test,y_pred,average = 'weighted')
    print("SVC f1: ", f1)

    # CNN model
    print('Trying a CNN + LSTM model ... ')
    
    # Define the model
    model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(1025,1)),  # 1D convolution layer
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')  # Use softmax for multi-class classification
])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=0,
                                       restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopper]
        )

    # Loss and accuracy
    print("accuracy:     ", history.history['accuracy'][-1])
    print("loss:         ", history.history['loss'][-1])
    print("val accuracy: ", history.history['val_accuracy'][-1])
    print("val loss:     ", history.history['val_loss'][-1])

    best_model = model
    return best_model
