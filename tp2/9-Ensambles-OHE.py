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

# +
import pandas as pd
import preprocessing as pp
import numpy as np
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import utils as utils
# -

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()


# - Se utilizan los mejores modelos obtenidos para XGBoost, Random Forest y SVM

def random_forest():
    preprocessor = pp.PreprocessingOHE()
    model = RandomForestClassifier(random_state=pp.RANDOM_STATE, 
                                   n_jobs=-1, 
                                   max_depth=8, 
                                   min_samples_leaf=1, 
                                   min_samples_split=14, 
                                   max_features=7)
    pipeline = Pipeline([("preprocessor", preprocessor), 
                         ("model", model)
                         ])
    return pipeline


def xgboost():
    pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingOHE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                            eval_metric="logloss", min_child_weight=6, max_depth=6, reg_alpha=0.05))
    ])
    return pipeline


def svm():
    preprocessor = pp.PreprocessingSE()
    model = SVC(kernel='rbf', random_state=pp.RANDOM_STATE, C=1, gamma='scale', probability=True)
    pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])
    return pipeline


def stacking_gaussian(var_smoothing=1e-9):
    estimadores = [('svm', svm()), ('xgboost', xgboost()), ('random_forest', random_forest())]
    cv = utils.kfold_for_cross_validation()
    stacking = StackingClassifier(estimators=estimadores, final_estimator=GaussianNB(var_smoothing=var_smoothing), 
                                  stack_method="predict_proba", cv=cv)
    return stacking


# ### Métricas finales

# Se eligió el [Modelo 1](#Modelo-1), dado que es el modelo que tiene mejores métricas en general (especialmente en Recall y F1 Score). En cuanto al Roc Auc, son los 3 muy parecidos.

pipeline = stacking_gaussian()

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# Este ensamble logra la mejor métrica ROC-AUC entre todos los modelos, a pesar de ser bastante similar a las obtenidas mediante 1-ArbolDeDecision y 2-RandomForest. Con la diferencia en que es el modelo con mejor Recall, debido a que la tasa de Falsos Negativos esta 2 puntos por debajo del Arbol de Decision y 6 por debajo de Random Forest. Sin embargo, obtiene peores resultados en cuanto a los Falsos Positivos, por lo cual no obtiene mejor Precision que dichos modelos.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/9-Ensambles-OHE.csv')


