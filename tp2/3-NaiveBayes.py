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
from sklearn.naive_bayes import CategoricalNB, GaussianNB
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()

# ### Modelo 1

# - Se utilizan únicamente las variables categóricas genero, tipo_sala y nombre_sede para realizar la clasificación
# - Se probó agregando las columnas `edad_isna` y `fila_isna` pero el modelo no mejoró sus resultados.

pipeline_1 = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB1()), 
                     ("model", CategoricalNB())
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline_1)

# ### Modelo 2

# - Se transforman las variables numéricas (precio_ticket y edad) en bins para poder utilizar solamente CategoricalNB.
# - Se realizan las mismas transformaciones que en el modelo anterior sobre las variables categóricas.
# - Se eliminaron las variables amigos y parientes debido a que no mejoraban el score del modelo.

pipeline_2 = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB2()), 
                     ("model", CategoricalNB())
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline_2)

# ### Modelo 3

# - Se utilizan unicamente las variables continuas y discretas
# - Se usa un GaussianNB

pipeline_3 = Pipeline([("preprocessor", pp.PreprocessingGaussianNB1()), 
                     ("model", GaussianNB())
                     ])

# #### Metricas

utils.metricas_cross_validation(X, y, pipeline_3)

# ### Modelo 4

# - Se combina un CategoricalNB con un GaussianNB usando un GaussianNB que toma la salida de los dos modelos anteriores para realizar la predicción. Para ello se hace un ensamble de tipo Stacking.
# - Se buscan mejores hiperparametros que los default con un GridSearchCV para ambos NB

# #### Hiperparámetros

pipeline_gaussian = Pipeline([("preprocessor", pp.PreprocessingGaussianNB1()), 
                              ("model", GaussianNB())
                     ])
pipeline_categorical = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB1()), 
                              ("model", CategoricalNB())
                     ])

# +
from sklearn.model_selection import GridSearchCV
params = {'model__alpha': np.arange(1, 10, 1)}

cv = utils.kfold_for_cross_validation()
#Descomentar para ejecutar GridSearchCV
gscv_categorical = GridSearchCV(pipeline_categorical, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True).fit(X, y)
print(gscv_categorical.best_score_)
print(gscv_categorical.best_params_)

# +
params = {'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.3]}

cv = utils.kfold_for_cross_validation()
#Descomentar para ejecutar GridSearchCV
gscv_gaussian = GridSearchCV(pipeline_gaussian, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True).fit(X, y)
print(gscv_gaussian.best_score_)
print(gscv_gaussian.best_params_)
# -

pipeline_gaussian = Pipeline([("preprocessor", pp.PreprocessingGaussianNB1()), 
                              ("model", GaussianNB(var_smoothing=0.01))
                     ])
pipeline_categorical = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB1()), 
                              ("model", CategoricalNB(alpha=2))
                     ])

# +
from sklearn.ensemble import StackingClassifier

estimadores = [('categorical_nb', pipeline_categorical), ('gaussian_nb', pipeline_gaussian)]
cv = utils.kfold_for_cross_validation()

stacked_naive_bayes = StackingClassifier(estimators=estimadores, final_estimator=GaussianNB(), stack_method="predict_proba", cv=cv)
# -

# #### Metricas

utils.metricas_cross_validation(X, y, stacked_naive_bayes)

# ### Métricas finales

# Se eligió el modelo que utiliza un ensamble de Stacking dado que, si bien el CV dió un poco peor que en el primer modelo del notebook, la diferencia es despreciable. Además al ser un ensamble, el algoritmo puede generalizar mejor.

# +
pipeline_gaussian = Pipeline([("preprocessor", pp.PreprocessingGaussianNB1()), 
                              ("model", GaussianNB(var_smoothing=0.01))
                     ])
pipeline_categorical = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB1()), 
                              ("model", CategoricalNB(alpha=2))
                     ])
estimadores = [('categorical_nb', pipeline_categorical), ('gaussian_nb', pipeline_gaussian)]
cv = utils.kfold_for_cross_validation()

stacked_naive_bayes = StackingClassifier(estimators=estimadores, final_estimator=GaussianNB(), stack_method="predict_proba", cv=cv)
# -

stacked_naive_bayes = utils.entrenar_y_realizar_prediccion_final_con_metricas(X, y, stacked_naive_bayes)

# La métrica objetivo AUC-ROC no superó la barrera de 0.90 obtenida en los modelos anteriores basados en arboles. Esto es causa de la alta tasa de Falsos Negativos obtenida por el modelo (7 puntos por encima de la obtenida por RandomForest) lo que afecta a todas las métricas a excepción del Accuracy.

# ### Predicción HoldOut

utils.predecir_holdout_y_generar_csv(stacked_naive_bayes, 'Predicciones/3-NaiveBayes.csv')
