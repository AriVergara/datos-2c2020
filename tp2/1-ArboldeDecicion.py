# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
import dtreeviz.trees as dtreeviz
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# ### Carga de Datasets

df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

# ### Preprocesamiento
# Con el metodo `procesamiento_arboles_discretizer` mejoran todas las metricas excepto `Precision`.

X_train, X_test, y_train, y_test = pp.procesamiento_arboles(df)

# ### Entrenamiento

clf = tree.DecisionTreeClassifier(random_state=117, max_depth=5, min_samples_leaf=4)
clf.fit(X_train, y_train)

# +
viz = dtreeviz.dtreeviz(
    clf,
    X_train,
    y_train,
    target_name='volveria',
    feature_names=list(X_train.columns),
    scale=1.5,
)

display(viz)

# +
max_depths = np.arange(1, 25)
min_samples_leafs = np.arange(1, 51)
data_points = []
for max_depth in max_depths:
    for min_samples_leaf in min_samples_leafs:
        clf_test = tree.DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=117
        )
        clf_test.fit(X_train, y_train)
        data_points.append(
            (max_depth, min_samples_leaf, accuracy_score(y_test, clf_test.predict(X_test)),)
        )

data_points = pd.DataFrame(
    data_points, columns=["max_depth", "min_samples_leaf", "score"]
)
plt.figure(dpi=125, figsize=(12, 8))
g = sns.heatmap(
    data_points.pivot_table(
        index="max_depth", columns="min_samples_leaf", values="score"
    ),
    square=True,
    cbar_kws=dict(use_gridspec=False, location="bottom"),
)
# -

# ### Metricas

y_pred = clf.predict(X_test)

# ##### AUC-ROC

round(roc_auc_score(y_test, y_pred), 3)

# ##### Accuracy

round(accuracy_score(y_test, y_pred), 2)

# ##### Precision

round(precision_score(y_test, y_pred), 2)

# ##### Recall

round(recall_score(y_test, y_pred), 2)

# ##### F1 Score

round(f1_score(y_test, y_pred), 2)

# ### Predicción

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir.head()


