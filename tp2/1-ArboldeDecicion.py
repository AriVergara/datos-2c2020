# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
import dtreeviz.trees as dtreeviz
import numpy as np
from ipywidgets import Button, IntSlider, interactive
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate

# ### Carga de Datasets

df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

df.head()

# ### Preprocesamiento
# Con el metodo `procesamiento_arboles_discretizer` mejoran todas las metricas excepto `Precision`.

y_train = df.volveria
X_train = pp.procesamiento_arboles(df.drop('volveria', axis=1, inplace=False))

X_train.head()

y_train.head()

model = tree.DecisionTreeClassifier(random_state=117, max_depth=4, min_samples_leaf=15)

# ### Métricas

cv = StratifiedKFold(n_splits=8, random_state=117, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring_metrics)

# ##### AUC-ROC

round(scores_for_model['test_roc_auc'].mean(), 3)

# ##### Accuracy

round(scores_for_model['test_accuracy'].mean(), 3)

# ##### Precision

round(scores_for_model['test_precision'].mean(), 3)

# ##### Recall

round(scores_for_model['test_recall'].mean(), 3)

# ##### F1 Score

round(scores_for_model['test_f1'].mean(), 3)

# ### Entrenamiento

model.fit(X_train, y_train)

# ### Predicción

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

df_predecir.head()

y_pred = model.predict(pp.procesamiento_arboles(df_predecir))

y_pred

# ### Gráfico árbol de decisión

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















# +
from sklearn.feature_selection import RFECV
min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2, shuffle=True, random_state=200),
              scoring='roc_auc',
              min_features_to_select=min_features_to_select)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()
# -

rfecv.


