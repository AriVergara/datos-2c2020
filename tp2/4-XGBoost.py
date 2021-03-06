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

# ### Modelo 1

# - Label encoder para las categóricas
# - Hiperparámetros por defecto (se setean dos para que no tire warnings)

# Como primera aproximación, se utiliza el preprocesador utilizado en Random Forest (que usa Label Encoding para las variables categóricas) dado que este modelo también se encuentra basado en árboles. Se utilizan los parámetros por deafault.

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingOHE()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 2

# - Se utiliza OHE para las categoricas
# - Se imputan los missings con la mediana en la edad
# - Se separa en dos bins la edad y el precio de ticket (se probó y da mejores resultados que no haciendolo).

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingXGBoost()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 3

# - No se completan los Nans, se deja que XGBoost se encargue de imputarlos

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingXGBoost2()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline)

# ### Modelo 4

# - Con el Modelo 1, se corre Grid Search para buscar los mejores hiperparametros

# Tuvimos un problema con este GridSearchCV. Por algún motivo, se quedaba estancado un largo rato en cada iteración. Para una grilla de tamaño 1 tardaba más de 10 minutos cuando entrenar el modelo por separado y aplicarle cross_validate tardaba un segundo. 
#
# Por ello se probaron a mano distintas configuraciones y se dejo la que mejor resultado obtuvo

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingOHE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                            eval_metric="logloss", min_child_weight=6, max_depth=6, reg_alpha=0.05))
])

utils.metricas_cross_validation(X, y, pipeline)

# +
params = {
    'model__learning_rate': [0.05, 0.1, 0.3],
    'model__max_depth': [3, 6, 10],
    'model__n_estimators': [100, 300],
    'model__min_child_weight': [1, 3, 5],
    'model__gamma': [0, 0.1, 0.2],
    'model__eval_metric': ['logloss', 'error']
}

cv = utils.kfold_for_cross_validation()
#gscv = GridSearchCV(pipeline, params, scoring='roc_auc', n_jobs=-1, cv=8, return_train_score=True).fit(X, y)
# -

# ### Métricas finales

# Se eligió el [Modelo 4](#Modelo-4) en base a los resultados obtenidos mediante `cross_validation`.

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingOHE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                            eval_metric="logloss", min_child_weight=6, max_depth=6, reg_alpha=0.05))
])

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# La métrica objetivo AUC-ROC tiene un resultado similar al obtenido por los modelos basados en arboles. Por el momento esto indica que este tipo de modelos obtienen una menor tasa de Falsos Negativos, mejorando todas las metricas que dependen de ello. Sin embargo, la tasa de Falsos Positivos de este modelo es un poco mayor que la obtenida en 2-RandomForest, por lo cual no logra obtener mejores métricas que dicho modelo.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/4-XGBoost.csv')
