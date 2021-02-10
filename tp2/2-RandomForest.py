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


def preprocesar_le(X_train, y_train, X_to_preprocess):
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
    
    encoder_nombre_sede = LabelEncoder()
    encoder_nombre_sede.fit(X_train['nombre_sede'].astype(str))
    X_to_preprocess['nombre_sede'] = encoder_nombre_sede.transform(X_to_preprocess['nombre_sede'].astype(str))
    
    encoder_nombre_sede = LabelEncoder()
    encoder_nombre_sede.fit(X_train['tipo_de_sala'].astype(str))
    X_to_preprocess['tipo_de_sala'] = encoder_nombre_sede.transform(X_to_preprocess['tipo_de_sala'].astype(str))
    
    encoder_nombre_sede = LabelEncoder()
    encoder_nombre_sede.fit(X_train['genero'].astype(str))
    X_to_preprocess['genero'] = encoder_nombre_sede.transform(X_to_preprocess['genero'].astype(str))
    
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
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self.mean_edad = 0
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)

        X['nombre_sede'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero.transform(X['genero'].astype(str))

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
                     ("model", RandomForestClassifier(n_jobs=-1))
                     ])

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=117, stratify=y, shuffle=True)

pipeline.fit(X_train, y_train)

roc_auc_score(pipeline.predict(X_test), y_test)


