# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import preprocesing as pp
import keras
from sklearn import preprocessing, tree
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder
)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from keras.wrappers.scikit_learn import KerasClassifier

import random
seed = 100
np.random.seed(seed)
random.seed(seed)
#import tensorflow as tf
#tf.set_random_seed(seed)

df_volvera = pd.read_csv('tp-2020-2c-train-cols1.csv')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('tp-2020-2c-train-cols2.csv')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

X = df.drop(columns="volveria", axis=1, inplace=False)
y = df["volveria"]

# ### Modelo 1

# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con OneHotEncoding
# - Red Neuronal de 3 capas, función de activación `tanh` para capas intermedias y `sigmoid` para capa final.
# - Optimizador `Adam`.

preprocessor = pp.PreprocessingSE()


def RedDeDosCapas():
    model = Sequential()
    model.add(Dense(8, input_dim=14, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


model = KerasClassifier(RedDeDosCapas, epochs=600, verbose=0)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 2

# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con OneHotEncoding
# - Red Neuronal de 5 capas, función de activación `relu` para capas intermedias y `sigmoid` para capa final.
# - GridSearchCV para busqueda de estimador.

preprocessor = pp.PreprocessingSE()


def redDeDosCapas_2(optimizer='Adam'):
    model = Sequential()
    model.add(Dense(8, input_dim=15, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(redDeDosCapas_2, epochs=600, verbose=0)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# +
from sklearn.model_selection import GridSearchCV
params = {'model__optimizer': optimizer}

cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)
# -

gscv.best_score_

gscv.best_params_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# +
#y_train = keras.utils.to_categorical(y_train, 2)
#y_test = keras.utils.to_categorical(y_test, 2)
# -

# - 3 Capas (8, 4 y 4 neuronas)
# - Opt: Adamax
# - epoch: 1000
# - activacion: relu

# + jupyter={"source_hidden": true}
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
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

# - 2 Capas (8, 4 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: tanh
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
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

# - 2 Capas (8, 4 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: tanh

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
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

# - 3 Capas (8, 4, 4 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: relu

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
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



# - 3 Capas (8, 4, 4 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: relu
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
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

# - 3 Capas (8, 4, 4 neuronas)
# - EarlyStopping patience 50
# - Opt: Adam
# - epoch: 600
# - activacion: relu
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
custom_early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, mode='max')
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=600, 
    validation_data=(preprocessor.transform(X_test), y_test), verbose=0, callbacks=[custom_early_stopping]
)

# + jupyter={"source_hidden": true}
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
# -



# - 2 Capas (32, 16 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: tanh
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(32, input_dim=14, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(16, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=800, 
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

# - 2 Capas (32, 16 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: tanh
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(32, input_dim=14, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(16, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(
    preprocessor.fit_transform(X_train), y_train, epochs=800, 
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

# - 3 Capas (64, 32, 32 neuronas)
# - Dropout 0.20
# - Opt: Adam lr=0.0001
# - epoch: 600
# - activacion: relu

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
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



# - 3 Capas (8, 4, 4 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: relu
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
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



# - 3 Capas (8, 4, 4 neuronas)
# - Opt: SGD
# - epoch: 600
# - activacion: rleu
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
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



# - 3 Capas (8, 4, 4 neuronas)
# - Opt: Adamax
# - epoch: 600
# - activacion: relu
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
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



# - 3 Capas (8, 4, 4 neuronas)
# - Opt: SGD lr=0.01
# - epoch: 600
# - activacion: rleu
# - regularizacion L2

# + jupyter={"source_hidden": true}
from keras.regularizers import l2
model = Sequential()
model.add(Dense(8, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(4, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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



# - 6 Capas (128, 128, 64, 32, 16 neuronas)
# - Opt: SGD lr=0.001
# - epoch: 600
# - activacion: rleu
# - regularizacion L2

from keras.regularizers import l2
model = Sequential()
model.add(Dense(128, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
opt = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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

# - 5 Capas (128, 128, 64, 32, 16 neuronas)
# - Opt: Adam
# - epoch: 600
# - activacion: rleu
# - regularizacion L2

from keras.regularizers import l2
model = Sequential()
model.add(Dense(128, input_dim=14, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="sigmoid"))
opt = keras.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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


