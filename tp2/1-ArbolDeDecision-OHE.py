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

# ### Metricas finales

# Se eligió el [Modelo 3](#Modelo-3) en base a los resultados obtenidos mediante `cross_validation`.

preprocessor = pp.PreprocessingOHE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE, 
                               max_depth=13, 
                               min_samples_leaf=6, min_samples_split=2,max_features=6)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# Como puede verse, la métrica objetivo AUC-ROC tiene un buen resultado en este modelo. Lo que no se logra es un buen resultado de Recall y eso puede verse también en la matriz de confusión: De los casos verdaderamente positivos el modelo selecciona como negativos al 24%, esa tasa de Falsos Negativos perjudica el resultado de todas las métricas, pero principalmente al Recall (recordando que `Recall = TP / (TP + FN)`).

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/1-ArbolDeDecision-OHE.csv')


