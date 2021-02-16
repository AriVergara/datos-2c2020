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
import keras
from sklearn import preprocessing, tree
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.base import BaseEstimator, TransformerMixin

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
# - Perceptron lineal de sklearn

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingSE()),
    ("model", Perceptron(random_state=pp.RANDOM_STATE))
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

# El mal resultado se puede deber a que los datos no son linealmente separables.

# ### Modelo 2

# - Se utiliza el mismo preprocesamiento que en el modelo anterior
# - Se buscan hiperparámetros para ver si mejora el score, de lo contrario se descarta el modelo.

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingSE()),
    ("model", Perceptron(random_state=pp.RANDOM_STATE, n_jobs=-1))
])

# +
from sklearn.model_selection import GridSearchCV
params = {
    'model__penalty': ["elasticnet"],
    'model__alpha': [0.0001, 0.001, 0.00001],
    'model__l1_ratio': [0, 0.15, 0.4, 0.5, 0.3, 1],
    'model__max_iter': [1000, 2000],
    'model__early_stopping': [True, False],
    'model__n_iter_no_change': [5, 30, 60],
    'model__eta0': [1, 0.9, 0.5, 1.2]
}

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True, refit=True
).fit(X, y)
print(gscv.best_score_)
print(gscv.best_params_)
# -

gscv.best_estimator_

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingSE()),
    ("model", Perceptron(random_state=pp.RANDOM_STATE, 
                         n_jobs=-1,
                         alpha=0.0001,
                         early_stopping=True,
                         n_iter_no_change=30,
                         l1_ratio=0.3,
                         max_iter=1000,
                         penalty='elasticnet',
                         eta0=0.9
                        ))
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=pp.RANDOM_STATE, stratify=y, shuffle=True)

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingSE()),
    ("model", Perceptron(random_state=pp.RANDOM_STATE, 
                         n_jobs=-1,
                         alpha=0.0001,
                         early_stopping=True,
                         n_iter_no_change=30,
                         l1_ratio=0.3,
                         max_iter=1000,
                         penalty='elasticnet',
                         eta0=0.9
                        ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.decision_function(X_test)

scores = [accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = [roc_auc_score(y_test, y_pred_proba)]
results += [s(y_test, y_pred) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())

# Los resultados obtenidos fueron muy malos por lo que se abandonó el modelo.


