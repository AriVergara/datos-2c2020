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

class PreprocessingCategoricalNB1(BaseEstimator, TransformerMixin):
    """
        -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket) y con valores 
            continuos o discretos(parientes, amigos, edad y precio_ticket).
        -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
        -Agrega columnas edad_isna, fila_isna, va_con_amigos, va_con_parientes
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self._moda_nombre_sede = ""
    
    def fit(self, X, y=None):
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        self._moda_nombre_sede = self._obtener_moda_nombre_sede(X)
        return self

    def transform(self, X):
        X = X.copy()
        X['tipo_de_sala_encoded'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        X = X.drop(columns=["tipo_de_sala"], axis=1, inplace=False)
        
        X['genero_encoded'] = self.le_genero.transform(X['genero'].astype(str))
        X = X.drop(columns=["genero"], axis=1, inplace=False)
        
        X["nombre_sede"] = X["nombre_sede"].fillna(self._moda_nombre_sede)
        X['nombre_sede_encoded'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        X = X.drop(columns=["nombre_sede"], axis=1, inplace=False)
        
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["amigos"], axis=1, inplace=False)
        X = X.drop(columns=["parientes"], axis=1, inplace=False)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        
        return X
    
    def _obtener_moda_nombre_sede(self, X):
        return X.nombre_sede.value_counts().index[0]


pipeline_1 = Pipeline([("preprocessor", PreprocessingCategoricalNB1()), 
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

class PreprocessingCategoricalNB2(BaseEstimator, TransformerMixin):
    """
        -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket) y con valores 
            continuos o discretos(parientes, amigos, edad y precio_ticket).
        -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
        -Agrega columnas va_con_amigos, va_con_parientes
    """
    def __init__(self):
        super().__init__()
        self._valores_fila = []
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
    
    def fit(self, X, y=None):
        self._valores_fila = X.fila.dropna().unique()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        self._moda_nombre_sede = self._obtener_moda_nombre_sede(X)
        return self

    def transform(self, X):
        X = X.copy()
        X['tipo_de_sala_encoded'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        X = X.drop(columns=["tipo_de_sala"], axis=1, inplace=False)
        
        X['genero_encoded'] = self.le_genero.transform(X['genero'].astype(str))
        X = X.drop(columns=["genero"], axis=1, inplace=False)
        
        X["nombre_sede"] = X["nombre_sede"].fillna(self._moda_nombre_sede)
        X['nombre_sede_encoded'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        X = X.drop(columns=["nombre_sede"], axis=1, inplace=False)
        
        X['edad_bins'] = X['edad'].apply(self._bins_segun_edad_cuantiles)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        
        X['precio_ticket_bins'] = X['precio_ticket'].apply(self._bins_segun_precio)
        X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["amigos"], axis=1, inplace=False)
        X = X.drop(columns=["parientes"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        
        return X
    
    def reemplazar_valores_de_fila_desconocidos(self, fila):
        if fila not in self._valores_fila:
            return np.nan()
        return fila
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3
    
    def _bins_segun_edad(self, edad): 
        if np.isnan(edad):
            return 0
        if edad <= 18:
            return 1
        if 18 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        if 40 < edad <= 70:
            return 4
        return 5
    
    def _bins_segun_edad_cuantiles(self, edad):
        if np.isnan(edad):
            return 0
        if edad <= 23:
            return 1
        if 23 < edad <= 31:
            return 2
        if 31 < edad <= 41:
            return 3
        return 4
    
    def _obtener_moda_nombre_sede(self, X):
        return X.nombre_sede.value_counts().index[0]


pipeline_2 = Pipeline([("preprocessor", PreprocessingCategoricalNB2()), 
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

class PreprocessingGaussianNB1(BaseEstimator, TransformerMixin):
    """
        - Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket) y con valores 
            categoricos (genero, fila, tipo_de_sala, nombre_sede).
        - Se agrega la columna acompaniantes y se eliminan parientes y amigos.
    """
    def __init__(self):
        super().__init__()
        self._mean_edad = 0
    
    def fit(self, X, y=None):
        self._mean_edad = X["edad"].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["edad"] = X["edad"].fillna(self._mean_edad)
        X["acompaniantes"] = X["parientes"] + X["amigos"]
        
        X = X.drop(columns=["parientes"], axis=1, inplace=False)
        X = X.drop(columns=["amigos"], axis=1, inplace=False)
        X = X.drop(columns=["genero"], axis=1, inplace=False)
        X = X.drop(columns=["tipo_de_sala"], axis=1, inplace=False)
        X = X.drop(columns=["nombre_sede"], axis=1, inplace=False)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        
        return X


pipeline_3 = Pipeline([("preprocessor", PreprocessingGaussianNB1()), 
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

pipeline_gaussian = Pipeline([("preprocessor", PreprocessingGaussianNB1()), 
                              ("model", GaussianNB())
                     ])
pipeline_categorical = Pipeline([("preprocessor", PreprocessingCategoricalNB1()), 
                              ("model", CategoricalNB())
                     ])

# +
from sklearn.ensemble import StackingClassifier

estimadores = [('categorical_nb', pipeline_categorical), ('gaussian_nb', pipeline_gaussian)]
cv = StratifiedKFold(n_splits=2, random_state=pp.RANDOM_STATE, shuffle=True)

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

# ### Metricas finales

# Se eligió el modelo que utiliza un ensamble de Stacking dado que, si bien el CV dió un poco peor que en el primer modelo del notebook, la diferencia es despreciable. Además al ser un ensamble, el algoritmo puede generalizar mejor.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=pp.RANDOM_STATE, stratify=y)

# +
pipeline_gaussian = Pipeline([("preprocessor", PreprocessingGaussianNB1()), 
                              ("model", GaussianNB())
                     ])
pipeline_categorical = Pipeline([("preprocessor", PreprocessingCategoricalNB1()), 
                              ("model", CategoricalNB())
                     ])
estimadores = [('categorical_nb', pipeline_categorical), ('gaussian_nb', pipeline_gaussian)]
cv = StratifiedKFold(n_splits=2, random_state=pp.RANDOM_STATE, shuffle=True)

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


