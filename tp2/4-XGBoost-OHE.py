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
from xgboost.sklearn import XGBClassifier
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()

# ### Métricas finales

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingOHE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                            eval_metric="logloss", min_child_weight=6, max_depth=6, reg_alpha=0.05))
])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# La métrica objetivo AUC-ROC tiene un resultado similar al obtenido al utilizar LE. Sin embargo, se observa que aumento la tasa de Falsos Negativos con respecto al otro modelo, por lo que su Recall (y por ende su F1 Score) disminuyó (en 0.09). A su vez, mejoró levemente la tasa de Verdaderos Negativos. 

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/4-XGBoost-OHE.csv')
