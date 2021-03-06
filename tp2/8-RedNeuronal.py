# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import preprocessing as pp
from sklearn.model_selection import train_test_split
import utils as utils
import keras
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

import random
seed = 100
np.random.seed(seed)
random.seed(seed)
import tensorflow
tensorflow.random.set_seed(seed)
import os
os.environ['PYTHONHASHSEED']=str(seed)

# + jupyter={"source_hidden": true}
X, y = utils.importar_datos()

# + jupyter={"source_hidden": true}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

# + jupyter={"source_hidden": true}
preprocessor = pp.PreprocessingSE()


# -

# ### Modelo 1

# - Preprocesamiento con StandarScaler
# - 3 Capas (8, 4 y 4 neuronas)
# - Optimizador Adamax
# - Función activación capas intermedias: `relu`
# - Función activación capa final: `sigmoid`

# + jupyter={"source_hidden": true}
def red_1():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    return model


# -

# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_1()
history = model.fit(
        preprocessor.fit_transform(X_train), y_train, epochs=1000, 
        validation_data=(preprocessor.transform(X_test), y_test), verbose=0
    )

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -

# #### Métricas

# + jupyter={"source_hidden": true}
pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_1, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 2

# - Preprocesamiento con StandarScaler
# - 2 Capas (8 y 4 neuronas)
# - Optimizador Adam
# - Función activación capas intermedias: `tanh`
# - Función activación capa final: `sigmoid`
# - Regularización L2

def red_2():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(4, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# #### Gráfico accuracy vs epoch

model = red_2()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()

# #### Métricas

pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_2, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 3

# - Preprocesamiento con StandarScaler
# - 2 Capas (8 y 4 neuronas)
# - Optimizador Adam
# - Función activación capas intermedias: `tanh`
# - Función activación capa final: `sigmoid`

# + jupyter={"source_hidden": true}
def red_3():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# -

# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_3()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -

# #### Métricas

# + jupyter={"source_hidden": true}
pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_3, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 4

# - Preprocesamiento con StandarScaler
# - 3 Capas (8, 4 y 4 neuronas)
# - Optimizador Adam
# - Función activación capas intermedias: `relu`
# - Función activación capa final: `sigmoid`

# + jupyter={"source_hidden": true}
def red_4():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# -

# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_4()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# + jupyter={"source_hidden": true}
pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_4, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 5

# - Preprocesamiento con StandarScaler
# - 3 Capas (8, 4 y 4 neuronas)
# - Optimizador Adam
# - Función activación capas intermedias: `relu`
# - Función activación capa final: `sigmoid`
# - Regularización L2

# + jupyter={"source_hidden": true}
def red_5():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# -

# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_5()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -

# #### Métricas

# + jupyter={"source_hidden": true}
pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_5, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 6

# - Preprocesamiento con StandarScaler
# - 2 Capas (32 y 16 neuronas)
# - Optimizador Adam
# - Función activación capas intermedias: `tanh`
# - Función activación capa final: `sigmoid`
# - Regularización L2

# + jupyter={"source_hidden": true}
def red_6():
    model = Sequential()
    model.add(Dense(32, input_dim=14, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(16, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# -

# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_6()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -

# #### Métricas

# + jupyter={"source_hidden": true}
pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_6, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 7

# - Preprocesamiento con StandarScaler
# - 3 Capas (64, 32, 32 neuronas)
# - Dropout 0.25
# - Optimizador Adam con lr=0.0001
# - Función activación capas intermedias: `relu`
# - Función activación capa final: `sigmoid`
# - Regularización L2

# + jupyter={"source_hidden": true}
def red_7():
    model = Sequential()
    model.add(Dense(64, input_dim=14, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(1, activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# -

# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_7()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -
# #### Métricas

# + jupyter={"source_hidden": true}
pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_7, epochs=600, verbose=0))
                     ])

# + jupyter={"source_hidden": true}
scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)


# -

# ### Modelo 8

# - Preprocesamiento con StandarScaler
# - 3 Capas (8, 4 y 4 neuronas)
# - Optimizador SGD
# - Función activación capas intermedias: `relu`
# - Función activación capa final: `sigmoid`
# - Regularización L2

def red_8():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model


# #### Gráfico accuracy vs epoch

# + jupyter={"source_hidden": true}
model = red_8()
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -
# #### Métricas

pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_8, epochs=600, verbose=0))
                     ])

scores = utils.metricas_cross_validation_con_cross_validate(X, y, pipeline)

# ### Metricas finales

# Se eligió el Modelo 8 en base a los resultados obtenidos mediante `cross_validation`.

pipeline = Pipeline([("preprocessor", pp.PreprocessingSE()), 
                     ("model", KerasClassifier(red_8, epochs=600, verbose=0))
                     ])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# Se obtiene una buena métrica objetivo AUC-ROC, pero no se logra mejorar los resultados de Recall. La matriz de confusión muesta un resultado muy similar al obtenido por 6-KNN, el modelo obtiene una alta tasa de Falsos Negativos, calificando como que no volvería al 33% de los encuestados que sí volverían. Las tasas de FP, FN, TP y TN son similares a las obtenidas por 3-NaiveBayes y 6-KNN por lo también se obtienen métricas ROC-AUC similares.

# Aclaración: Se setearon todos los seeds que se encontraron en la documentación de Keras para lograr resultados reproducibles, sin embargo no se pudo lograr que las metricas no varien en las distintas ejecuciones.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/8-RedNeuronal.csv')


