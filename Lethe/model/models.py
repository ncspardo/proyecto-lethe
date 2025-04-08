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

def model_selection(X_train, y_train, X_test, y_test):
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

    best_model = model
    return best_model
