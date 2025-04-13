import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.pipeline import Pipeline
import pickle
from Lethe.params import *

# Función para aplicar FFT
def apply_fft(df, n_features=1024, frecuencia_muestreo=512):
    fft_features = []
    for index, row in df.iterrows():
        señal = row.iloc[:n_features].values.astype(float)
        fft_values = np.fft.fft(señal)
        fft_magnitudes = np.abs(fft_values)[:n_features//2]
        fft_features.append(fft_magnitudes)
    return np.array(fft_features)

def fft_prep(base_path, pfile):
    # Rutas a los archivos
    file_paths = {
        'healthy': 'bal_healthy.csv',
        'narco': 'bal_narco.csv',
        'ins': 'bal_ins.csv',
        'sdb': 'bal_sdb.csv'
        #'plm': 'bal_plm.csv'
        #'rbd': 'bal_rbd.csv'
        #'nfle': 'bal_nfle.csv'
    }

    # Etiquetas para diagnóstico
    diagnostic_labels = {
        'healthy': 0,
        'narco': 1,
        'ins': 2,
        'sdb': 3
        #'plm': 4
        #'nfle': 2
        #'rbd': 3
    }

    # Cargar datasets
    dfs = []
    n_features = 1024
    columns_features = [f'Feature_{i+1}' for i in range(n_features)]

    for key, filename in file_paths.items():
        df = pd.read_csv(base_path + "/" + filename, header=None)
        df.rename(columns={1024: 'Sleep_stages'}, inplace=True)
        df.columns = columns_features + ['Sleep_stages']
        df['Diagnostic'] = diagnostic_labels[key]
        dfs.append(df)

    # Concatenar todos
    df = pd.concat(dfs, ignore_index=True)

    transformer = FunctionTransformer(apply_fft)
    pipeline = Pipeline ([
        ('fft', transformer),
        ('scaler', StandardScaler())
    ])

    # FFT
    X_fft = pipeline.fit_transform(df)

    # Save the pipeline
    with open(pfile, 'wb') as file:
        pickle.dump(pipeline, file)

    X = np.hstack((X_fft, df['Sleep_stages'].values.reshape(-1, 1)))

    # Features + Sleep_stages
    y = df['Diagnostic'].values

    # Convertir etiquetas a formato one-hot
    y_one_hot = to_categorical(y, num_classes=4)  # Cambié a 7 clases

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y
    )
    return [X_train, y_train, X_test, y_test, X_test, y_test]

def fft_model(X_train, y_train, X_test, y_test, X_val, y_val):

    # Modelo de Red Neuronal
    model = Sequential()

    # Capa de entrada con 512 nodos y activación ReLU
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))

    # Capa oculta con 256 nodos y activación ReLU
    model.add(Dense(256, activation='relu'))

    # Capa oculta con 128 nodos y activación ReLU
    model.add(Dense(128, activation='relu'))

    # Capa oculta con 64 nodos y activación ReLU
    model.add(Dense(64, activation='relu'))

    # Regularización: Dropout con tasa de 0.5
    model.add(Dropout(0.5))

    # Capa de salida con 7 nodos (uno por cada clase) y activación softmax
    model.add(Dense(4, activation='softmax'))

    # Compilación del modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    # Early stopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Entrenamiento
    print("Entrenando modelo de Red Neuronal...")
    model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1, validation_split=0.2)

    # Evaluación del modelo
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convertir las probabilidades a clases

    # Convertir etiquetas de test a clases
    y_test_classes = np.argmax(y_test, axis=1)

    # Evaluación
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'\nAccuracy del modelo de Red Neuronal: {accuracy:.4f}')

    print("\nReporte de clasificación:")
    print(classification_report(y_test_classes, y_pred_classes, digits=4))

    return model
