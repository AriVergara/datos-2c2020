# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
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
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

# +
from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': np.arange(1, 31), 'model__min_samples_leaf': np.arange(1, 16),
         "n_estimators": 200, "min_samples_split":2}

gscv = GridSearchCV(
    pipeline, params, scoring='accuracy', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -

gscv.best_params_

model = RandomForestClassifier(random_state=pp.RANDOM_STATE, 
                               n_jobs=-1, 
                               max_depth=12, 
                               min_samples_leaf=4)

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

# ### Metricas finales

# Se eligió el....

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

preprocessor = pp.PreprocessingOHE()
model = tree.DecisionTreeClassifier(random_state=pp.RANDOM_STATE)

pipeline = Pipeline([("preprocessor", preprocessor), 
                     ("model", model)
                     ])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

scores = [roc_auc_score, accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = [s(y_pred, y_test) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())

# ### Predicción HoldOut

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir['volveria'] = pipeline.predict(df_predecir)
df_predecir = df_predecir[['id_usuario', 'volveria']]

#with open('1-ArbolDeDecision.csv', 'w') as f:
 #   df_predecir.to_csv(f, sep=',', index=False)


# ### Modelo 0

# +
def clasificar_encuestado(fila):
    if fila['edad'] < 18:
        acompaniantes = fila['parientes'] + fila['amigos']
        return 1 if acompaniantes <= 3 else 0
    if fila['genero'] == 'hombre':
        return 0
    if fila['tipo_de_sala'] == '4d' and fila['nombre_sede'] == 'fiumark_palermo':
        return 0
    return 1


def baseline(X):
    resultado = []
    for nro_fila in range(len(X)):
        resultado.append(clasificar_encuestado(df.loc[nro_fila,:]))
    return resultado


# -

y_pred_baseline = baseline(X_test)

scores = [roc_auc_score, accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
results = [s(y_pred_baseline, y_test) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())

















# +
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class PreprocessingLE(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        super().__init__()
        self.le_tipo_sala_ = LabelEncoder()
        self.le_nombre_sede_ = LabelEncoder()
        self.le_genero_ = LabelEncoder()
        self.mean_edad_ = 0
    
    def fit(self, X, y=None):
        self.mean_edad_ = X["edad"].mean()
        self.le_tipo_sala_.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede_.fit(X['nombre_sede'].astype(str))
        self.le_genero_.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad_)

        X['nombre_sede'] = self.le_nombre_sede_.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala_.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero_.transform(X['genero'].astype(str))

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3


# -

pipeline = Pipeline([("preprocessor", PreprocessingTransformer()), 
                     ("model", RandomForestClassifier(random_state=117, n_jobs=-1))
                     ])

# +
from sklearn.model_selection import GridSearchCV
params = {'model__max_depth': np.arange(1, 31), 'model__min_samples_leaf': np.arange(1, 16)}

gscv = GridSearchCV(
    pipeline, params, scoring='accuracy', n_jobs=-1, cv=5, return_train_score=True
).fit(X, y)
# -



print(f"Best score: {gscv.best_score_}")
print(f"Best params {gscv.best_params_}")

X = df.drop(columns="volveria", axis=1, inplace=False)
y = df["volveria"]


def random_forest_cv(X, y, preprocesar, rf_params={}, cv_n_splits=8, random_state=117):
    kf = StratifiedKFold(n_splits=cv_n_splits, random_state=random_state, shuffle=True)

    test_accuracies = []
    test_roc_aucs = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        clf = RandomForestClassifier(n_jobs=-1, random_state=random_state, **rf_params)

        X_train_cv = X.loc[train_index,]
        y_train_cv = y[train_index]
        X_test_cv = X.loc[test_index,]
        y_test_cv = y[test_index]

        X_train_cv_preprocesado = preprocesar(X_train_cv, y_train_cv, X_train_cv)
        clf.fit(X_train_cv_preprocesado, y_train_cv)

        X_test_cv_preprocesado = preprocesar(X_train_cv, y_train_cv, X_test_cv)
        y_predict_cv = clf.predict(X_test_cv_preprocesado)

        test_roc_auc = roc_auc_score(y_test_cv, y_predict_cv)
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


random_forest_cv(X=X, 
                 y=y, 
                 preprocesar=preprocesar_le, 
                 rf_params={"max_depth": 100, "min_samples_leaf":10, "n_estimators": 200, "min_samples_split":2})

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

# +
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        super().__init__()
        self.le_tipo_sala_ = LabelEncoder()
        self.le_nombre_sede_ = LabelEncoder()
        self.le_genero_ = LabelEncoder()
        self.mean_edad_ = 0
    
    def fit(self, X, y=None):
        self.mean_edad_ = X["edad"].mean()
        self.le_tipo_sala_.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede_.fit(X['nombre_sede'].astype(str))
        self.le_genero_.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad_)

        X['nombre_sede'] = self.le_nombre_sede_.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala_.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero_.transform(X['genero'].astype(str))

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3


# -

pipeline = Pipeline([("preprocessor", PreprocessingTransformer()), 
                     ("model", RandomForestClassifier(random_state=117, n_jobs=-1))
                     ])

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=117, stratify=y)

# +
#pipeline.fit(X_train, y_train)

# +
#y_pred = pipeline.predict(X_test)
#roc_auc_score(pipeline.predict(X_test), y_test)
#print(f"mean test roc auc is: {roc_auc_score(y_pred, y_test):.4f}")
#print(f"mean test accuracy is: {accuracy_score(y_pred, y_test):.4f}")
#print(f"mean test precision is: {precision_score(y_pred, y_test):.4f}")
#print(f"mean test recall is: {recall_score(y_pred, y_test).mean():.4f}")
#print(f"mean test f1_score is: {f1_score(y_pred, y_test):.4f}")
# -

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
scores_for_model

print(f"mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")


def our_own_cv(X, y, clf, cv_n_splits=8, random_state=117):
    kf = StratifiedKFold(n_splits=cv_n_splits, random_state=random_state, shuffle=True)

    test_accuracies = []
    test_roc_aucs = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    X = X.reset_index().drop(columns=['index'])
    y = y.reset_index().drop(columns=['index'])['volveria']
    display(y)
    display(X)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        y_test_cv = y[test_index]
        
        clf.fit(X.loc[train_index,], y[train_index])
        y_predict_cv = clf.predict(X.loc[test_index,])

        test_roc_auc = roc_auc_score(y_test_cv, y_predict_cv)
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


our_own_cv(X=X_train, 
                 y=y_train, 
                 clf=pipeline)




