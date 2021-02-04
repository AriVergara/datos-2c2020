# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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

# + jupyter={"source_hidden": true}
import pandas as pd
import keras
import preprocesing as pp
from sklearn import preprocessing, tree
import dtreeviz.trees as dtreeviz
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
# -

# ### Carga de Datasets

df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')



from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)



# ### Preprocesamiento

X_train, X_test, y_train, y_test = pp.procesamiento_arboles(df)

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train)

X_train.columns

clf.feature_importances_



# ### Entrenamiento

model = Sequential()
model.add(Dense(8, input_dim=len(X_train.columns), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation="sigmoid"))

#opt = keras.optimizers.RMSprop(lr=0.001)
#opt = keras.optimizers.SGD(learning_rate=0.001)
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

#y_train = keras.utils.to_categorical(y_train, 2)
#y_test = keras.utils.to_categorical(y_test, 2)
y_train.shape

X_train.values.shape

model.output_shape

history = model.fit(
    X_train.values, y_train, epochs=600, validation_data=(X_test.values, y_test), verbose=0
)

fig = plt.figure(figsize=(12, 6), dpi=100)
plt.ylabel("Accuracy")
plt.xlabel("epoc")
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()

# ### Metricas

y_pred = (model.predict(X_test) > 0.5).astype("int32")

# ##### AUC-ROC

round(roc_auc_score(y_test, y_pred), 3)

# ##### Accuracy

round(accuracy_score(y_test, y_pred), 2)

# ##### Precision

round(precision_score(y_test, y_pred), 2)

# ##### Recall

round(recall_score(y_test, y_pred), 2)

# ##### F1 Score

round(f1_score(y_test, y_pred), 2)

# ### Predicción

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir.head()


