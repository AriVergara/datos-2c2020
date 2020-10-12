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

# %%
