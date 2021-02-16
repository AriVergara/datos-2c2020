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
import preprocesing as pp
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder
)
from sklearn.naive_bayes import CategoricalNB, GaussianNB
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier

# +
import random
seed = 100
np.random.seed(seed)
random.seed(seed)

#When using tensorflor
#import tensorflow as tf
#tf.set_random_seed(seed)
# -

df_volvera = pd.read_csv('tp-2020-2c-train-cols1.csv')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('tp-2020-2c-train-cols2.csv')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

X = df.drop(columns="volveria", axis=1, inplace=False)
y = df["volveria"]

# ### Modelo 1

# - Se utilizan únicamente las variables categóricas genero, tipo_sala y nombre_sede para realizar la clasificación

pipeline_1 = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB1()), 
                     ("model", CategoricalNB())
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline_1, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 2

# - Se transforman las variables numéricas (precio_ticket y edad) en bins para poder utilizar solamente CategoricalNB.
# - Se realizan las mismas transformaciones que en el modelo anterior sobre las variables categóricas.
# - Se eliminaron las variables amigos y parientes debido a que no mejoraban el score del modelo.

pipeline_2 = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB2()), 
                     ("model", CategoricalNB())
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline_2, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 3

# - Se utilizan unicamente las variables continuas y discretas
# - Se usa un GaussianNB

pipeline_3 = Pipeline([("preprocessor", pp.PreprocessingGaussianNB1()), 
                     ("model", GaussianNB())
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline_3, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

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

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
gscv_categorical = GridSearchCV(
    pipeline_categorical, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)
print(gscv_categorical.best_score_)
print(gscv_categorical.best_params_)

# +
params = {'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 0.1, 0.3]}

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
gscv_gaussian = GridSearchCV(
    pipeline_gaussian, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)
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
cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)

stacked_naive_bayes = StackingClassifier(estimators=estimadores, final_estimator=GaussianNB(), stack_method="predict_proba", cv=cv)
# -

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(stacked_naive_bayes, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Métricas finales

# Se eligió el modelo que utiliza un ensamble de Stacking dado que, si bien el CV dió un poco peor que en el primer modelo del notebook, la diferencia es despreciable. Además al ser un ensamble, el algoritmo puede generalizar mejor.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=pp.RANDOM_STATE, stratify=y)

# +
pipeline_gaussian = Pipeline([("preprocessor", pp.PreprocessingGaussianNB1()), 
                              ("model", GaussianNB(var_smoothing=0.01))
                     ])
pipeline_categorical = Pipeline([("preprocessor", pp.PreprocessingCategoricalNB1()), 
                              ("model", CategoricalNB(alpha=2))
                     ])
estimadores = [('categorical_nb', pipeline_categorical), ('gaussian_nb', pipeline_gaussian)]
cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)

stacked_naive_bayes = StackingClassifier(estimators=estimadores, final_estimator=GaussianNB(), stack_method="predict_proba", cv=cv)
# -

stacked_naive_bayes.fit(X_train, y_train)

y_pred = stacked_naive_bayes.predict(X_test)
y_pred_proba = stacked_naive_bayes.predict_proba(X_test)[:, 1]

scores = [accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = [roc_auc_score(y_test, y_pred_proba)]
results += [s(y_test, y_pred) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())

# ### Predicción HoldOut

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir['volveria'] = stacked_naive_bayes.predict(df_predecir)
df_predecir = df_predecir[['id_usuario', 'volveria']]

with open('Predicciones/3-NaiveBayes.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)


