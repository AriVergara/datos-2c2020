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
import utils as utils
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()

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

cv = utils.kfold_for_cross_validation()
#rgscv = RandomizedSearchCV(
#    pipeline, params, n_iter=50, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(rgscv.best_score_)
#print(rgscv.best_params_)
# -

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale', probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 2

# - Kernel Polinomico
# - Preprocesamiento con StandardScaler
# - Estimación de Hiperparametros mediante RandomSearch
# - Preprocesamiento de variables categoricas con OneHotEncoding

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='poly', random_state=pp.RANDOM_STATE, probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
params = {'model__C': np.arange(1, 150, 10), 'model__degree': np.arange(1, 3)}

cv = utils.kfold_for_cross_validation()
#rgscv = RandomizedSearchCV(
#    pipeline, params, n_iter=30, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(rgscv.best_score_)
#print(rgscv.best_params_)
# -

model = SVC(kernel='poly', random_state=pp.RANDOM_STATE, C=1, degree=2, probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
params = {'model__gamma': ['scale', 'auto'], 'model__coef0': np.arange(1, 150, 25)}

cv = utils.kfold_for_cross_validation()
#rgscv = GridSearchCV(
#    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(rgscv.best_score_)
#print(rgscv.best_params_)
# -

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='poly', random_state=pp.RANDOM_STATE, C=1, degree=2, 
            gamma='scale', coef0=101, probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 3 

# - Kernel Lineal
# - Estimación de Hiperparametros con GridSearchCV
# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con OneHotEncoding

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='linear', random_state=pp.RANDOM_STATE, probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
from sklearn.model_selection import GridSearchCV
params = {'model__C': np.arange(1, 250, 10)}

cv = utils.kfold_for_cross_validation()
#gscv = GridSearchCV(
#    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(gscv.best_score_)
#print(gscv.best_params_)

# +
params = {'model__C': np.arange(30, 60)}

cv = utils.kfold_for_cross_validation()
#gscv = GridSearchCV(
#    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(gscv.best_score_)
#print(gscv.best_params_)
# -

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='linear', random_state=pp.RANDOM_STATE, C=30, probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 4

# - Kernel Radial
# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con LabelEncoding

preprocessor = pp.PreprocessingSE_2()
model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale', probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Métricas finales

# Dado que el Modelo 1 (kernel radial) y el Modelo 2 (kernel polinomico) obtuvieron resultados similares mediante `cross_validation`, se optó por elegir el [Modelo 1](#Modelo-1).

preprocessor = pp.PreprocessingSE()
model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale', probability=True)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# La métrica objetivo AUC-ROC de este modelo no supera a las obtenidas por los modelos basados en arboles. Es este caso, se debe a la tasa de Falsos Positivos obtenidas por el modelo, la cual afecta a casi todas las métricas pero principalmente a Precision, por eso es el modelo que obtiene el peor resultado en ese apartado hasta el momento. En cuanto a la tasa de True Positive y False Negative el resultado obtenido es igual al de 4-XGBoost.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/5-SVM.csv')
