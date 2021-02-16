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
from sklearn import tree
import numpy as np
import seaborn as sns
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()

# ### Modelo 1

# - Preprocesamiento con LaberEncoding
# - Hiperparametros por defecto

preprocessor = pp.PreprocessingLE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 2

# - Preprocesamiento con OneHotEncoding
# - Hiperparametros por defecto

preprocessor = pp.PreprocessingOHE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 3 

# - Preprocesamiento con LabelEncoder
# - Estimación de Hiperparametros con GridSearchCV

preprocessor = pp.PreprocessingLE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': [10, 20, 50, None], 'model__min_samples_leaf': [1, 5, 10, 15, 20],
          "model__min_samples_split": [2, 5, 10, 15], "model__criterion": ["gini", "entropy"], 
          "model__max_features": ["auto", "log2", 7, 2]}
cv = utils.kfold_for_cross_validation()
gscv = GridSearchCV(pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True).fit(X, y)
print(gscv.best_params_)
print(gscv.best_score_)

from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': np.arange(10,25), 'model__min_samples_leaf': np.arange(3,10),
         "model__min_samples_split": np.arange(1,7), 
          "model__max_features": ["auto", "log2"]+list(np.arange(5,10)),
         "model__criterion": ["gini", "entropy"]}
cv = utils.kfold_for_cross_validation()
gscv = GridSearchCV(pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True).fit(X, y)
print(gscv.best_params_)
print(gscv.best_score_)

model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE, 
                               max_depth=13, 
                               min_samples_leaf=6, min_samples_split=2,max_features=6)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

utils.metricas_cross_validation(X, y, pipeline)

# ### Metricas finales

# Se eligió el [Modelo 3](#Modelo-3) en base a los resultados obtenidos mediante `cross_validation`.

preprocessor = pp.PreprocessingLE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE, 
                               max_depth=13, 
                               min_samples_leaf=6, min_samples_split=2,max_features=6)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/1-ArbolDeDecision.csv')
