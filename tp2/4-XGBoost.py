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

class PreprocessingLE(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad con la media
    -Convierte en bins los valores de las columnas edad y precio_ticket.
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self.mean_edad = 0
        self.moda_nombre_sede = ""
    
    def fit(self, X, y=None):
        self.moda_nombre_sede = X["nombre_sede"].astype(str).mode()[0]
        self.mean_edad = X["edad"].mean()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].fillna(self.moda_nombre_sede).astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        X["edad_bins"] = X["edad"].apply(self._bins_segun_edad_2)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        
        X["nombre_sede_isna"] = X["nombre_sede"].isna().astype(int)
        X['nombre_sede'] = X['nombre_sede'].fillna(self.moda_nombre_sede)
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
    
    def _bins_segun_edad(self, edad): 
        if edad <= 20:
            return 1
        if 20 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        return 4
    
    def _bins_segun_edad_2(self, edad): 
        if edad <= 18:
            return 1
        if 18 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        if 40 < edad <= 70:
            return 4
        return 5


pipeline = Pipeline([
    ("preprocessor", PreprocessingLE()),
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

pipeline = Pipeline([
    ("preprocessor", PreprocessingLE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", eval_metric="logloss", n_estimators=50, learning_rate=0.2, njobs=-1))
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

# ### Modelo 4

# - Con el Modelo 1, se corre Grid Search para buscar los mejores hiperparametros

# Tuvimos un problema con este GridSearchCV. Por algún motivo, se quedaba estancado un largo rato en cada iteración. Para una grilla de tamaño 1 tardaba más de 10 minutos cuando entrenar el modelo por separado y aplicarle cross_validate tardaba un segundo. 
#
# Por ello se probaron a mano distintas configuraciones y se dejo la que mejor resultado obtuvo

pipeline = Pipeline([
    ("preprocessor", PreprocessingLE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                            eval_metric="logloss", min_child_weight=6, max_depth=6, reg_alpha=0.05))
])

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
scores_for_model = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics)
print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}")
print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}")
print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}")
print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}")
print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}")

# +
params = {
    'model__learning_rate': [0.05, 0.1, 0.3],
    'model__max_depth': [3, 6, 10],
    'model__n_estimators': [100, 300],
    'model__min_child_weight': [1, 3, 5],
    'model__gamma': [0, 0.1, 0.2],
    'model__eval_metric': ['logloss', 'error']
}

cv = StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)
gscv = GridSearchCV(
    pipeline, params, scoring='roc_auc', n_jobs=-1, cv=8, return_train_score=True
).fit(X, y)
# -

# ### Metricas finales

# Se eligió el [Modelo 1](#Modelo-1) en base a los resultados obtenidos mediante `cross_validation`.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)

pipeline = Pipeline([
    ("preprocessor", PreprocessingLE()),
    ("model", XGBClassifier(use_label_encoder=False, scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", n_estimators=1000, learning_rate=0.01, n_jobs=-1,
                            eval_metric="logloss", min_child_weight=6, max_depth=6, reg_alpha=0.05))
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

with open('Predicciones/4-XGBoost.csv', 'w') as f:
    df_predecir.to_csv(f, sep=',', index=False)
