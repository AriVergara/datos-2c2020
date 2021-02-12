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
from sklearn import preprocessing, tree
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder
)
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

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

# - Preprocesamiento con LaberEncoding
# - Hiperparametros por defecto

preprocessor = pp.PreprocessingLE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# ### Modelo 2

# - Preprocesamiento con OneHotEncoding
# - Hiperparametros por defecto

preprocessor = pp.PreprocessingOHE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# #### Metricas

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

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
cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)

gscv.best_params_

gscv.best_score_

from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': np.arange(10,25), 'model__min_samples_leaf': np.arange(3,10),
         "model__min_samples_split": np.arange(1,7), 
          "model__max_features": ["auto", "log2"]+list(np.arange(5,10)),
         "model__criterion": ["gini", "entropy"]}
cv = StratifiedKFold(n_splits=5, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=cv, return_train_score=True
).fit(X, y)

gscv.best_params_

gscv.best_score_

model = RandomForestClassifier(random_state=pp.RANDOM_STATE, 
                               n_jobs=-1, 
                               max_depth=11, 
                               min_samples_leaf=4, min_samples_split=2,max_features=5)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")
print(scores_for_model['test_roc_auc'])


def our_own_cv(X, y, clf, cv_n_splits=8, random_state=117):
    kf = StratifiedKFold(n_splits=cv_n_splits, random_state=random_state, shuffle=True)

    test_accuracies = []
    test_roc_aucs = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        y_test_cv = y[test_index]
        
        clf.fit(X.loc[train_index,], y[train_index])
        y_predict_cv = clf.predict(X.loc[test_index,])

        test_roc_auc = roc_auc_score(y_test_cv, clf.predict_proba(X.loc[test_index,])[:, 1])
        test_roc_aucs.append(test_roc_auc)

        test_accuracy = accuracy_score(y_test_cv, y_predict_cv)
        test_accuracies.append(test_accuracy)

        test_precision = precision_score(y_test_cv, y_predict_cv)
        test_precisions.append(test_precision)

        test_recall = recall_score(y_test_cv, y_predict_cv)
        test_recalls.append(test_recall)

        test_f1_score = f1_score(y_test_cv, y_predict_cv)
        test_f1_scores.append(test_f1_score)

    print(f"mean test roc auc is: {np.mean(test_roc_aucs):.4f}")
    print(f"mean test accuracy is: {np.mean(test_accuracies):.4f}")
    print(f"mean test precision is: {np.mean(test_precisions):.4f}")
    print(f"mean test recall is: {np.mean(test_recalls):.4f}")
    print(f"mean test f1_score is: {np.mean(test_f1_scores):.4f}")
    print(test_roc_aucs)


our_own_cv(X, y, pipeline, cv_n_splits=8, random_state=pp.RANDOM_STATE)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
scores_for_model = cross_val_score(pipeline, X, y, cv=cv, scoring=make_scorer(roc_auc_score))

scores_for_model.mean()

# ### Metricas finales

# Se eligió el [Modelo 3](#Modelo-3) en base a los resultados obtenidos mediante `cross_validation`.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

preprocessor = pp.PreprocessingLE()
model = RandomForestClassifier(random_state=pp.RANDOM_STATE, 
                               n_jobs=-1, 
                               max_depth=11, 
                               min_samples_leaf=4, min_samples_split=2,max_features=5)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

scores = [accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = [roc_auc_score(y_test, y_pred_proba)]
results += [s(y_test, y_pred) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())

# ### Predicción HoldOut

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir['volveria'] = pipeline.predict(df_predecir)
df_predecir = df_predecir[['id_usuario', 'volveria']]

with open('1-ArbolDeDecision.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)


