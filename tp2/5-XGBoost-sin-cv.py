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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb

# ### Carga de Datasets

df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

# ### Preprocesamiento

X_train, X_test, y_train, y_test = pp.procesamiento_arboles_discretizer(df)

# ### Entrenamiento

# +
#train_matrix = xgb.DMatrix(X_train, label=y_train)
#test_matrix = xgb.DMatrix(X_test, label=y_test)
#eval_list = [(test_matrix, "eval"), (train_matrix, "train")]
#num_rounds = 10
#params = {'eval_metric': 'auc'}
#model = xgb.train(params, train_matrix, num_rounds, eval_list)
#y_pred = model.predict(test_matrix)
# -

model = xgb.XGBClassifier(objective='binary:logistic', )
model.fit(X_train, y_train)

# ### Metricas

y_pred = model.predict(X_test)

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


