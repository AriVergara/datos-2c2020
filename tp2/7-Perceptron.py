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
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()

# ### Modelo 1

# - Preprocesamiento con StandardScaler
# - Preprocesamiento de variables categoricas con OneHotEncoding
# - Perceptron lineal de sklearn

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingSE()),
    ("model", Perceptron(random_state=pp.RANDOM_STATE))
])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline, True)

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

cv = utils.kfold_for_cross_validation()
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

utils.metricas_cross_validation(X, y, pipeline, True)

# ### Metricas finales

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

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline, True)

# Los resultados obtenidos fueron muy malos por lo que se abandonó el modelo.
