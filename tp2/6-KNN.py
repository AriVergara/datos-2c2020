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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
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

# - Preprocesamiento con StandardScaler
# - Estimación de Hiperparametros con GridSearchCV
# - Estimación de algortimo con GridSearchCV

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

params = {'model__n_neighbors': np.arange(1, 50, 5), 'model__weights': ['uniform', 'distance'], 
          'model__algorithm': ['ball_tree', 'kd_tree', 'brute']}
cv = utils.kfold_for_cross_validation()
#rgscv = GridSearchCV(
#    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(rgscv.best_score_)
#print(rgscv.best_params_)

# +
from sklearn.neighbors import KDTree

params = {'model__n_neighbors': np.arange(1, 50, 5), 'model__weights': ['uniform', 'distance'], 
          'model__metric': KDTree.valid_metrics}
cv = utils.kfold_for_cross_validation()
#rgscv = GridSearchCV(
#    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(rgscv.best_score_)
#print(rgscv.best_params_)
# -

model = KNeighborsClassifier(n_jobs=-1, n_neighbors=21, algorithm='kd_tree', weights='uniform', metric='manhattan')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

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
cv = utils.kfold_for_cross_validation()
#gscv = GridSearchCV(
#    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)

# +
#gscv.best_score_

# +
#gscv.best_params_
# -

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1, algorithm='ball_tree', n_neighbors=21, weights='uniform', metric='canberra')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Metricas finales

# Se eligió el [Modelo 1](#Modelo-1) en base a los resultados obtenidos mediante `cross_validation`.

preprocessor = pp.PreprocessingSE()
model = KNeighborsClassifier(n_jobs=-1, n_neighbors=21, algorithm='kd_tree', weights='uniform', metric='manhattan')

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# Se obtiene una buena métrica objetivo AUC-ROC, pero no se logra mejorar los resultados de Recall. Nuevamente el modelo obtiene una alta tasa de Falsos Negativos, calificando como que no volvería al 33% de los encuestados que sí volverían. La cantidad de Falsos Positivos es baja por lo cual no se ve reducido significativamente el AUC-ROC.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/6-KNN.csv')
