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
import preprocesing as pp
from sklearn import preprocessing
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder
)
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import utils as utils

# +
import random
seed = 100
np.random.seed(seed)
random.seed(seed)

#When using tensorflor
#import tensorflow as tf
#tf.set_random_seed(seed)
# -

df_volvera = pd.read_csv('tp-2020-2c-train-cols1.csv')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('tp-2020-2c-train-cols2.csv')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

X = df.drop(columns="volveria", axis=1, inplace=False)
y = df["volveria"]

# ### Modelo 1

# - Preprocesamiento con LaberEncoding
# - Hiperparametros por defecto

preprocessor = pp.PreprocessingLE()
model = RandomForestClassifier(random_state=pp.RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 2

# - Preprocesamiento con OneHotEncoding
# - Hiperparametros por defecto

preprocessor = pp.PreprocessingOHE()
model = RandomForestClassifier(random_state=pp.RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 3 

# - Preprocesamiento con LabelEncoder
# - Estimación de Hiperparametros con GridSearchCV

preprocessor = pp.PreprocessingLE()
model = RandomForestClassifier(random_state=pp.RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': [10, 20, 50, None], 'model__min_samples_leaf': [1, 5, 10, 15, 20],
         "model__n_estimators": [50, 100, 400], "model__min_samples_split": [2, 5, 10, 15], 
          "model__criterion": ["gini", "entropy"], "model__max_features": ["auto", "log2", 7, 2]}

cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True, refit=True
).fit(X, y)
# -

gscv.best_params_

gscv.best_score_

from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': np.arange(5,15), 'model__min_samples_leaf': np.arange(1,5),
         "model__n_estimators": [75, 100, 125], "model__min_samples_split": np.arange(12, 25)}
cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True, refit=True
).fit(X, y)

gscv.best_params_

gscv.best_score_

model = RandomForestClassifier(random_state=pp.RANDOM_STATE, 
                               n_jobs=-1, 
                               max_depth=11, 
                               min_samples_leaf=1, 
                               min_samples_split=13)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

utils.metricas_cross_validation(X, y, pipeline)

# ### Metricas finales

# Se eligió el....

X_train, X_test, y_train, y_test = utils.train_test_split_para_prediccion_final(X, y)

preprocessor = pp.PreprocessingLE()
model = RandomForestClassifier(random_state=pp.RANDOM_STATE, 
                               n_jobs=-1, 
                               max_depth=11, 
                               min_samples_leaf=1, 
                               min_samples_split=13)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

utils.metricas_finales(y_test, y_pred, y_pred_proba)

# ### Predicción HoldOut

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir['volveria'] = pipeline.predict(df_predecir)
df_predecir = df_predecir[['id_usuario', 'volveria']]

with open('Predicciones/2-RandomForest.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)
