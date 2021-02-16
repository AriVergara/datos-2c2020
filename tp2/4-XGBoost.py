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
from xgboost.sklearn import XGBClassifier
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

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

# - Label encoder para las categóricas
# - Hiperparámetros por defecto (se setean dos para que no tire warnings)

# Como primera aproximación, se utiliza el preprocesador utilizado en Random Forest (que usa Label Encoding para las variables categóricas) dado que este modelo también se encuentra basado en árboles. Se utilizan los parámetros por deafault.

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingLE()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
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

# - Se utiliza OHE para las categoricas
# - Se imputan los missings con el promedio en la edad
# - Se separa en dos bins la edad y el precio de ticket (se probó y da mejores resultados que no haciendolo).

# +
from sklearn.preprocessing import KBinsDiscretizer

class PreprocessingXGBoost(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self._bins_edad = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile")
        self._bins_precio_ticket = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile")
    
    def fit(self, X, y=None):
        self._mediana_edad = X.edad.median()
        self._mediana_precio_ticket = X.precio_ticket.median()
        self._bins_edad.fit(X[["edad"]].fillna(X.edad.median()))
        self._bins_precio_ticket.fit(X[["precio_ticket"]].fillna(X.precio_ticket.median()))
        return self

    def transform(self, X):
        X = X.copy()
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self._mediana_edad)
        X["edad_bins"] = pd.DataFrame(self._bins_edad.transform(X[["edad"]]))[0]
        #X = X.drop(columns=["edad"], axis=1, inplace=False)
        #X["edad_limite_inferior"] = X["edad"].apply(self.setear_limite_inferior_bins, args=(self._bins_edad,))
        #X["edad_limite_superior"] = X["edad"].apply(self.setear_limite_superior_bins, args=(self._bins_edad,))
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)

        X["precio_ticket"] = X["precio_ticket"].fillna(self._mediana_precio_ticket)
        X["precio_ticket_bins"] = pd.DataFrame(self._bins_precio_ticket.transform(X[["precio_ticket"]]))[0]
        #X["precio_ticket_limite_inferior"] = X["precio_ticket"].apply(self.setear_limite_inferior_bins, args=(self._bins_precio_ticket,))
        #X["precio_ticket_limite_superior"] = X["precio_ticket"].apply(self.setear_limite_superior_bins, args=(self._bins_precio_ticket,))
        #X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        return X
    
    def setear_limite_inferior_bins(self, selected_bin, discretizer):
        limites = discretizer.bin_edges_[0]
        cantidad_bins = discretizer.n_bins_[0]
        for i in range(cantidad_bins):
            if limites[i] <= selected_bin < limites[i+1]:
                return limites[i]
            
    def setear_limite_superior_bins(self, selected_bin, discretizer):
        limites = discretizer.bin_edges_[0]
        cantidad_bins = discretizer.n_bins_[0]
        for i in range(cantidad_bins):
            if limites[i] <= selected_bin < limites[i+1]:
                return limites[i+1]
        return limites[cantidad_bins]
            


# -

pipeline = Pipeline([
    ("preprocessor", PreprocessingXGBoost()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
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

# - No se completan los Nans, se deja que XGBoost se encargue de imputarlos

# +
from sklearn.impute import SimpleImputer

class PreprocessingXGBoost2(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)
        
        return X


# -

pipeline = Pipeline([
    ("preprocessor", PreprocessingXGBoost2()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
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

# - Con el Modelo 2, se corre Grid Search para buscar los mejores hiperparametros

pipeline = Pipeline([
    ("preprocessor", pp.PreprocessingLE()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# +
params = {
    'model__learning_rate': [0.05, 0.1, 0.15, 0.2, 0.3],
    'model__max_depth': [3, 4, 5, 6, 8, 10],
    'model__n_estimators': [50, 100, 200, 300],
    'model__scale_pos_weight': [1],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8],
    'model__objective': ["binary:logistic"],
    'model__min_child_weight':range(1,6,2),
    'model__gamma': [0, 0.1, 0.2],
    'model__use_label_encoder': [False],
    'model__eval_metric': ['logloss', 'error', 'auc']
}

gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=8, return_train_score=True
).fit(X, y)
# -

# ### Metricas finales

# Se eligió el [Modelo 1](#Modelo-1) en base a los resultados obtenidos mediante `cross_validation`.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

pipeline = Pipeline([
    ("preprocessor", PreprocessingXGBoost()),
    ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
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

with open('Predicciones/6-KNN.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)























class OneHotEncoderWrapper(object):
    """
        Wrappea un OneHotEncoder para poder manejar de forma correcta los Nans y evitar la dummy trap.
        Si la columna presenta missings, no se dropea ninguna columna y se generan solo columnas para los valores correspondientes.
        De esta manera, un 0 en todas las columnas representa un missing o bien un valor no visto en el set de entrenamiento.
    """
    def __init__(self, columnas):
        super().__init__()
        self._most_frequent_imputer = SimpleImputer(strategy='most_frequent', fill_value=1)
        self._ohe = OneHotEncoder(handle_unknown="ignore", drop=None)
        self._columnas = columnas
        self._drop_first = False
    
    def fit(self, X, y=None):
        X = X.copy()
        X = X[self._columnas]
        X = self._most_frequent_imputer.fit_transform(X)
        if data[self._columna].isnull().sum() > 0:
            data = data.dropna().astype(str)
        else:
            self._drop_first = True
        self._ohe.fit(data)
        return self

    def transform(self, X):
        X = X.copy()
        column_transformed = pd.DataFrame(self._ohe.transform(
            X[['nombre_sede']].dropna().astype(str)).todense().astype(int)).add_prefix(f'{self._columna}_')
        X = pd.concat([X, column_transformed], axis=1)
        X = X.drop(columns=[self._columna], axis=1, inplace=False)
        if self._drop_first:
            X = X.drop(columns=[f'{self._columna}_0'], axis=1, inplace=False)
        return X


class OneHotEncoderWrapper(object):
    """
        Wrappea un OneHotEncoder para poder manejar de forma correcta los Nans y evitar la dummy trap.
        Si la columna presenta missings, no se dropea ninguna columna y se generan solo columnas para los valores correspondientes.
        De esta manera, un 0 en todas las columnas representa un missing o bien un valor no visto en el set de entrenamiento.
    """
    def __init__(self, columna):
        super().__init__()
        self._ohe = OneHotEncoder(handle_unknown="ignore", drop=None)
        self._columna = columna
        self._drop_first = False
    
    def fit(self, X, y=None):
        data = X[[self._columna]].copy()
        if data[self._columna].isnull().sum() > 0:
            data = data.dropna().astype(str)
        else:
            self._drop_first = True
        self._ohe.fit(data)
        return self

    def transform(self, X):
        X = X.copy()
        column_transformed = pd.DataFrame(self._ohe.transform(
            X[['nombre_sede']].fillna("").astype(str)).todense().astype(int)).add_prefix(f'{self._columna}_')
        X = pd.concat([X, column_transformed], axis=1)
        X = X.drop(columns=[self._columna], axis=1, inplace=False)
        if self._drop_first:
            X = X.drop(columns=[f'{self._columna}_0'], axis=1, inplace=False)
        return X


o = OneHotEncoderWrapper("genero")
o.fit(X)
o.transform(df_predecir)

o._ohe.categories_

df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')

s = SimpleImputer(strategy='most_frequent', fill_value=1)
o = OneHotEncoder(handle_unknown="ignore", drop=None)
o.fit(s.fit_transform(X[["nombre_sede", "tipo_de_sala"]]))

o.categories_

o.transform(X[["nombre_sede", "tipo_de_sala"]])


