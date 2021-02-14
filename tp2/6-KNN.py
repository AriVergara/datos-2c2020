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
from sklearn import preprocessing, tree
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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

# - Preprocesamiento con StandardScaler
# - Estimación de Hiperparametros con GridSearchCV
# - Estimación de algortimo con GridSearchCV

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
params = {'model__n_neighbors': np.arange(1, 50, 5), 'model__weights': ['uniform', 'distance'], 
          'model__algorithm': ['ball_tree', 'kd_tree', 'brute']}

rgscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -

rgscv.best_score_

rgscv.best_params_

# +
from sklearn.neighbors import KDTree

params = {'model__n_neighbors': np.arange(1, 50, 5), 'model__weights': ['uniform', 'distance'], 
          'model__metric': KDTree.valid_metrics}

rgscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -

rgscv.best_score_

rgscv.best_params_

model = KNeighborsClassifier(n_jobs=-1, n_neighbors=21, algorithm='kd_tree', weights='uniform', metric='manhattan')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 2

# - Algoritmo ball-tree
# - Preprocesamiento con StandardScaler
# - Estimación de metrica mediante GridSearchCV

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1, algorithm='ball_tree')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
from sklearn.neighbors import BallTree

params = {'model__n_neighbors': np.arange(1, 50, 5), 'model__weights': ['uniform', 'distance'], 
          'model__metric': BallTree.valid_metrics}

gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -

gscv.best_score_

gscv.best_params_

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1, algorithm='ball_tree', n_neighbors=21, weights='uniform', metric='canberra')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Metricas finales

# Se eligió el [Modelo 1](#Modelo-1) en base a los resultados obtenidos mediante `cross_validation`.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1, n_neighbors=21, algorithm='kd_tree', weights='uniform', metric='manhattan')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

scores = [accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = [roc_auc_score(y_test, y_pred_proba)]
results += [s(y_test, y_pred) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())

# ### Predicción HoldOut

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir['volveria'] = pipeline.predict(df_predecir)
df_predecir = df_predecir[['id_usuario', 'volveria']]

with open('4-KNN.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)


