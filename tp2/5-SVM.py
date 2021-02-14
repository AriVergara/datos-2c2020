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

# - Kernel Radial
# - Preprocesamiento con StandardScaler
# - Estimacion de Hiperparametros con RandomSearchCV
# - Preprocesamiento de variables categoricas con OneHotEncoding

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
params = {'model__C': np.arange(1, 150, 25), 'model__gamma': ['scale', 'auto'] + list(np.arange(1, 20))}

rgscv = RandomizedSearchCV(
    pipeline, params, n_iter=50, scoring='roc_auc', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -

rgscv.best_score_

rgscv.best_params_

model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale')

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

# - Kernel Polinomico
# - Preprocesamiento con StandardScaler
# - Estimación de Hiperparametros mediante RandomSearch
# - Preprocesamiento de variables categoricas con OneHotEncoding

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='poly', random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
params = {'model__C': np.arange(1, 150, 25), 'model__degree': np.arange(1, 5), 
          'model__gamma': np.arange(1, 150, 25), 'model__coef0': np.arange(1, 150, 25)}

rgscv = RandomizedSearchCV(
    pipeline, params, n_iter=10, scoring='roc_auc', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -

rgscv.best_score_

rgscv.best_params_

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 3 

# - Kernel Lineal
# - Estimación de Hiperparametros con GridSearchCV
# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con OneHotEncoding

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='linear', random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
from sklearn.model_selection import GridSearchCV
params = {'model__C': np.arange(1, 250, 10)}

cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)
# -

gscv.best_params_

gscv.best_score_

# +
params = {'model__C': np.arange(30, 60)}

cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)
# -

gscv.best_params_

gscv.best_score_

model = model = SVC(kernel='linear', random_state=pp.RANDOM_STATE, C=51)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 4

# - Kernel Radian
# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con LabelEncoding

preprocessor = pp.PreprocessingSE_2()
model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.150, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale', probability=True)

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

with open('3-SVM.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)


