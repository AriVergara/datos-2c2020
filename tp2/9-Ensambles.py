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


# ### Modelo 1

# - Se utiliza un ensamble de tipo Stacking
# - Como estimador final se usa un GaussianNB
#

def stacking_gaussian(var_smoothing=1e-9):
    estimadores = [('svm', svm()), ('xgboost', xgboost()), ('random_forest', random_forest())]
    cv = utils.kfold_for_cross_validation()
    stacking = StackingClassifier(estimators=estimadores, final_estimator=GaussianNB(var_smoothing=var_smoothing), 
                                  stack_method="predict_proba", cv=cv)
    return stacking


stacking = stacking_gaussian()

# #### Metricas

utils.metricas_cross_validation(X, y, stacking)

# ### Modelo 2

# - Mismo tipo de ensamble
# - Mismo estimador final que en el modelo 1, pero se busca un mejor hiperparámetro para el GaussianNB.
#

stacking = stacking_gaussian()

# +
#params = {
#    'final_estimator__var_smoothing': [1e-9, 1e-7, 1e-6, 1e-3, 5e-3, 1e-2, 0.1, 0.3],
#    'xgboost__model__use_label_encoder': [False], 
#    'xgboost__model__scale_pos_weight': [1], 
#    'xgboost__model__subsample': [0.8], 
#    'xgboost__model__colsample_bytree': [0.8],
#    'xgboost__model__objective': ["binary:logistic"], 
#    'xgboost__model__n_estimators': [1000], 
#    'xgboost__model__learning_rate': [0.01], 
#    'xgboost__model__n_jobs': [-1],                       
#    'xgboost__model__eval_metric': ["logloss"], 
#    'xgboost__model__min_child_weight': [6], 
#    'xgboost__model__max_depth': [6], 
#    'xgboost__model__reg_alpha': [0.05],
#    'svm__model__C': [1], 
#    'svm__model__gamma': ['scale'], 
#    'svm__model__probability': [True],
#    'svm__model__random_state': [pp.RANDOM_STATE], 
#    'random_forest__model__n_jobs': [-1], 
#    'random_forest__model__max_depth': [11], 
#    'random_forest__model__min_samples_leaf': [1], 
#    'random_forest__model__min_samples_split': [13]
#}

#cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
#gscv_gaussian = GridSearchCV(
#    stacking, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
#).fit(X, y)
#print(gscv_gaussian.best_score_)
#print(gscv_gaussian.best_params_)
# -

# Al igual que en el Notebook de XGBoost, el Grid Search tarda mucho en correr a pesar de que la grilla es pequeña (solo final_estimator__var_smoothing tiene más de un valor). Para evitar esto se prueba el var_smoothing a mano.

from sklearn.model_selection import cross_validate
options = [1e-9, 1e-8, 1e-7, 1e-6, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.3]
max_score_value = 0
optimal_var_smothing = 0
#for var_smothing in options:
#    stacking = stacking_gaussian(var_smothing)
#    cv = utils.kfold_for_cross_validation()
#    scoring_metrics = ["roc_auc"]
#    scores_for_model = cross_validate(stacking, X, y, cv=cv, scoring=scoring_metrics)
#    roc_auc_score_value = scores_for_model['test_roc_auc'].mean()
#    print(f"Corrio con var_smothing: {var_smothing}, roc_auc_score: {roc_auc_score_value}")
#    if roc_auc_score_value > max_score_value:
#        max_score_value = roc_auc_score_value
#        optimal_var_smothing = var_smothing
#print(f'var_smothing: {optimal_var_smothing}')
#print(f'roc_auc_score_value: {max_score_value}')

stacking = stacking_gaussian(0.03)

# #### Métricas

utils.metricas_cross_validation(X, y, stacking)


# ### Modelo 3

# - Ensamble de tipo Voting (soft voting).
# - Se utilizan los mismos modelos que en los ensambles de tipo Stacking

def voting_classifier(voting="soft"):
    estimadores = [('svm', svm()), ('xgboost', xgboost()), ('random_forest', random_forest())]
    stacking = VotingClassifier(estimators=estimadores, n_jobs=-1, voting=voting)
    return stacking


voting = voting_classifier()

utils.metricas_cross_validation(X, y, voting)

# ### Métricas finales

# Se eligió el [Modelo 1](#Modelo-1), dado que es el modelo que tiene mejores métricas en general (especialmente en Recall y F1 Score). En cuanto al Roc Auc, son los 3 muy parecidos.

pipeline = stacking_gaussian()

pipeline = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline)

# Este ensamble logra la mejor métrica ROC-AUC entre todos los modelos, a pesar de ser bastante similar a las obtenidas mediante 1-ArbolDeDecision y 2-RandomForest. Con la diferencia en que es el modelo con mejor Recall, debido a que la tasa de Falsos Negativos esta 2 puntos por debajo del Arbol de Decision y 6 por debajo de Random Forest. Sin embargo, obtiene peores resultados en cuanto a los Falsos Positivos, por lo cual no obtiene mejor Precision que dichos modelos.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(pipeline, 'Predicciones/9-Ensambles.csv')
