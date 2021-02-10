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
from sklearn import preprocessing
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# +
import random
seed = 100
np.random.seed(seed)
random.seed(seed)

#When using tensorflor
#import tensorflow as tf
#tf.set_random_seed(seed)
# -

df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')


def preprocesar(X_train, y_train, X_to_preprocess):
    def bins_segun_precio(valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3

    X_to_preprocess["fila_isna"] = X_to_preprocess["fila"].isna().astype(int)
    X_to_preprocess = X_to_preprocess.drop(columns=["fila"], axis=1, inplace=False)
    
    X_to_preprocess["edad_isna"] = X_to_preprocess["edad"].isna().astype(int)
    X_to_preprocess["edad"] = X_to_preprocess["edad"].fillna(X_train["edad"].mean())
    
    X_to_preprocess = pd.get_dummies(X_to_preprocess, columns=['nombre_sede'], dummy_na=True, drop_first=True)
    
    X_to_preprocess = pd.get_dummies(X_to_preprocess, columns=['tipo_de_sala'], drop_first=True, dummy_na=True)
    
    X_to_preprocess = pd.get_dummies(X_to_preprocess, columns=['genero'], drop_first=True, dummy_na=True)
    
    X_to_preprocess["precio_ticket_bins"] = X_to_preprocess["precio_ticket"].apply(bins_segun_precio)
    
    X_to_preprocess = X_to_preprocess.drop(columns=["id_usuario"], axis=1, inplace=False)
    X_to_preprocess = X_to_preprocess.drop(columns=["nombre"], axis=1, inplace=False)
    X_to_preprocess = X_to_preprocess.drop(columns=["id_ticket"], axis=1, inplace=False)
    
    return X_to_preprocess


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
        clf = RandomForestClassifier(random_state=random_state, **rf_params)

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
                 preprocesar=preprocesar, 
                 rf_params={"max_depth": 100, "min_samples_leaf":10, "n_estimators": 200, "min_samples_split":2})

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

model = RandomForestClassifier(max_depth=5, min_samples_leaf=3)
cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(model, preprocesar(X,y,X), y, cv=cv, scoring=scoring_metrics)

round(scores_for_model['test_roc_auc'].mean(), 3)


