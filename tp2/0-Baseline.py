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
import utils as utils
import numpy as np
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

import random
seed = 100
np.random.seed(seed)
random.seed(seed)

X, y = utils.importar_datos()

X[~(X["genero"] == "hombre") & ~(X["tipo_de_sala"] == "4d")].tipo_de_sala.value_counts()

# ### Modelo 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)


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

def _probabilidades_grupo(X_grupo):
    cantidad_elementos = len(X_grupo)
    cantidad_positivos = len(X_grupo[X_grupo["volveria"] == 1])
    prob_volveria = cantidad_positivos/float(cantidad_elementos)
    prob_no_volveria = (cantidad_elementos - cantidad_positivos)/float(cantidad_elementos)
    return [prob_no_volveria, prob_volveria]

def clasificar_encuestado_proba(fila, X):
    if fila['edad'] < 18:
        X_grupo = X[X["edad"] < 18]
        
        if fila['acompaniantes'] <= 3:
            X_grupo = X_grupo[X_grupo["acompaniantes"] <= 3]
            return _probabilidades_grupo(X_grupo)
        else:
            X_grupo = X_grupo[X_grupo["acompaniantes"] > 3]
            return _probabilidades_grupo(X_grupo)
    
    if fila['genero'] == 'hombre':
        X_grupo = X[X["genero"] == 'hombre']
        return _probabilidades_grupo(X_grupo)
    
    if fila['tipo_de_sala'] == '4d' and fila['nombre_sede'] == 'fiumark_palermo':
        X_grupo = X[(X["genero"] == 'mujer') & 
                    (X['tipo_de_sala'] == '4d') & 
                    (X['nombre_sede'] == 'fiumark_palermo')]
        return _probabilidades_grupo(X_grupo)
    
    X_grupo = X[(X["genero"] == 'mujer') & 
                ~(X['tipo_de_sala'] == '4d') & 
                ~(X['nombre_sede'] == 'fiumark_palermo')]
    return _probabilidades_grupo(X_grupo)

def baseline(X):
    resultado = []
    for indice in X.index:
        resultado.append(clasificar_encuestado(X.loc[indice,:]))
    return resultado

def baseline_proba(X):
    X = X.copy()
    X['acompaniantes'] = X['parientes'] + X['amigos']         
    resultado = []
    for indice in X.index:
        clasificacion = clasificar_encuestado_proba(X.loc[indice,:], X)
        resultado.append(clasificacion)
    return resultado


# +
y_pred_baseline = baseline(X_test)
X_test["volveria"] = y_test
y_pred_proba = baseline_proba(X_test)
y_pred_proba = np.array(y_pred_proba)[:, 1]

scores = [accuracy_score, precision_score, recall_score, f1_score]
columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

results = [roc_auc_score(y_test, y_pred_proba)]

results += [s(y_test, y_pred_baseline) for s in scores]
display(pd.DataFrame([results], columns=columnas).style.hide_index())
# -


