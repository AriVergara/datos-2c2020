# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP Parte 1 - Organización de Datos 2C 2020
# ## Importando librerias
#

# %%
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns

from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

# %% [markdown]
# ## Cargando DataSets
# Son dos datasets por separado, por lo cual el primer paso es cargarlos y hacer un merge.

# %% [markdown]
# El primer dataset contiene el `id_usuario`y el booleano `volveria`

# %%
df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1GoWOQS1bZSKIphmJkkLdVY-BIbTWatQS', index_col=0)
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_volvera

# %% [markdown]
# El segundo dataset contiene el resto de los datos encuestados

# %%
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1jyKSMAqjKZB_J90mcQJAGZyx_NCz2myJ', index_col=0)
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos

# %%
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

# %%
df.set_index('id_usuario', inplace=True)
df

# %%
from pandas_profiling import ProfileReport

report = ProfileReport(
    df, title='Resumen de datos TP Parte 1', explorative=True, lazy=False
)

# %%
report.to_file('reporte.html')

# %%
df.reset_index(inplace=True)

# %%
df.head()

# %% [markdown]
# Como se ve en el reporte de Pandas Profiling, el 77% de las filas no tiene un valor en la columna fila, por lo que decidimos que no era de importancia dada la cantidad de missing values que contiene.

# %%
df.drop(axis=1, columns=["fila"], inplace=True)

# %%
df.head()

# %%
df[df.duplicated(subset=df.columns.drop(["id_usuario", "volveria", "tipo_de_sala", "nombre"]))]

# %% [markdown]
# Se puede ver que si bien el id_ticket aparece repetido, el nombre de la persona que completo la encuesta es diferente, así como su id_usuario. Por lo que interpretamos que esta columna no nos aporta información para predecir la decisión del usuario

# %%
df.drop(axis=1, columns=["id_ticket"], inplace=True)

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
sns.countplot(x="tipo_de_sala", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Tipo de sala")
axes[0].set_title("Cantidad de encuestados segun tipo de sala y si volvería a ver Frozen 4")


sns.countplot(x="tipo_de_sala", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Tipo de sala")
axes[1].set_title("Cantidad de encuestados segun tipo de sala")
plt.show()

# %% [markdown]
# El unico tipo de sala que presenta una diferencia notoria es la de 4d, que a su vez es la que mas entradas presenta en el df.

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
sns.countplot(x="nombre_sede", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Sede")
axes[0].set_title("Cantidad de encuestados segun sede del cine y si volvería a ver Frozen 4")


sns.countplot(x="nombre_sede", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Sede")
axes[1].set_title("Cantidad de encuestados segun sede del cine")
plt.show()

# %% [markdown]
# Casi todos los datos provienen de la sede de palermo

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(6.4 * 2, 4.8))
palermo = df[df["nombre_sede"] == "fiumark_palermo"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=palermo, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Sede")
axes[0].set_title("Palermo")

quilmes = df[df["nombre_sede"] == "fiumark_quilmes"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=quilmes, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Sede")
axes[1].set_title("Quilmes")


chacarita = df[df["nombre_sede"] == "fiumark_chacarita"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=chacarita, ax=axes[2])
axes[2].set_ylabel("Cantidad")
axes[2].set_xlabel("Sede")
axes[2].set_title("Chacarita")



# %%
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
sns.countplot(x="genero", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Género")
axes[0].set_title("Cantidad de encuestados segun género y si volvería a ver Frozen 4")


sns.countplot(x="genero", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Género")
axes[1].set_title("Cantidad de encuestados segun género")
plt.show()

# %% [markdown]
# La mayoría de los hombres decide no volver mientras que con las mujeres ocurre lo contrario

# %%
plt.figure(dpi=100)
plt.title("Distribución del precio de acuerdo a si vuelve o no")
sns.boxplot(
    data=df[df.precio_ticket <= 10],
    y="precio_ticket",
    x="volveria",
)
plt.ylabel("Precio del ticket")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# %%
Genero y edad
Tipo de sala, cine y precio
Amigos y parientes

