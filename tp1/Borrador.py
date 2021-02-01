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
#from pandas_profiling import ProfileReport

#report = ProfileReport(
    #df, title='Resumen de datos TP Parte 1', explorative=True, lazy=False
#)

# %%
#report.to_file('reporte.html')

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
fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)
palermo = df[df["nombre_sede"] == "fiumark_palermo"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=palermo, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Sede")
axes[0].set_title("Palermo")


chacarita = df[df["nombre_sede"] == "fiumark_chacarita"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=chacarita, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Sede")
axes[1].set_title("Chacarita")

quilmes = df[df["nombre_sede"] == "fiumark_quilmes"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=quilmes, ax=axes[2])
axes[2].set_ylabel("Cantidad")
axes[2].set_xlabel("Sede")
axes[2].set_title("Quilmes")






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

# %% [markdown]
# # TODO (12/10)
# - Genero y edad
# - Tipo de sala, cine y precio
# - Amigos y parientes


# %% [markdown]
# ### Relacionando Genero y Edad

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de los encuestados de acuerdo a su genero")
sns.boxplot(
    data=df[df.precio_ticket <= 10],
    y="edad",
    x="genero",
)
plt.ylabel("Edad")
plt.show()

# %% [markdown]
# ## Relacionando Tipo de Sala, Sede y Precio

# %%
cooccurrence = pd.pivot_table(df, 
                              values="precio_ticket",
                              index="nombre_sede", 
                              columns="tipo_de_sala", 
                              aggfunc='mean')
plt.figure(dpi=100)
plt.title("Precio primedio pagado por la entrada", fontsize=9)
sns.heatmap(cooccurrence, square=True, cmap="Wistia")
plt.show()

# %% [markdown]
# ### Cantidad de acompañantes

# %% [markdown]
# Primero vamos a probar crear una nueva variable `acompañantes`para buscar relaciones entre las distintas variables y la cantidad de acompañantes con los que vieron la pelicula, sin importar de que tipo sean.

# %%
df['acompaniantes'] = df.parientes + df.amigos 

# %%
df.head()

# %%
plt.figure(dpi=100)
sns.countplot(
    x="acompaniantes", data=df, order=df["acompaniantes"].value_counts().index
)
plt.ylabel("Cantidad")
plt.xlabel("Numero de acompañantes")
plt.title("Cantidad de encuestados según numero de acompañantes")
plt.show()

# %%
plt.figure(dpi=100)
sns.countplot(x="acompaniantes", hue="volveria", data=df)
plt.ylabel("Cantidad")
plt.xlabel("Acompañantes")
plt.title("Cantidad de encuestados segun cantidad de acompañantes y si volvería a ver Frozen 4")
plt.show()

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
sns.countplot(x="acompaniantes", hue="volveria", data=df[df.genero == 'hombre'], ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Acompañantes")
axes[0].set_title('Hombres')

sns.countplot(x='acompaniantes', hue='volveria', data=df[df.genero=='mujer'], ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Acompañantes")
axes[1].set_title('Mujeres')
figs.suptitle("Cantidad de encuestados segun cantidad de acompañantes y si volveria a ver Frozen 4")
plt.show()

# %% [markdown]
# Cuando el hombre va acompañado la diferencia se achica? O es por las cantidades?

# %% [markdown]
# ### Dividiendo en familiares y amigos

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
sns.countplot(x="acompaniantes", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Amigos")

sns.countplot(x='acompaniantes', hue='volveria', data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Parientes")
figs.suptitle("Cantidad de encuestados segun numero y tipo de acompañantes")
plt.show()

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
sns.countplot(x="amigos", hue="volveria", data=df[df.genero == 'mujer'], ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Amigos")

sns.countplot(x='parientes', hue='volveria', data=df[df.genero=='mujer'], ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Parientes")
figs.suptitle("Cantidad de mujeres encuestadas segun cantidad de acompañantes y si volveria a ver Frozen 4")
plt.show()

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
sns.countplot(x="amigos", hue="volveria", data=df[df.genero == 'hombre'], ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Amigos")

sns.countplot(x='parientes', hue='volveria', data=df[df.genero=='hombre'], ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Parientes")
figs.suptitle("Cantidad de hombres encuestados segun cantidad de acompañantes y si volveria a ver Frozen 4")
plt.show()

# %% [markdown]
# Buscar relacion entre volveria y genero masculino  || no volveria y genero femenino

# %% [markdown]
# ### Relación genero y precio ticket

# %%
plt.figure(dpi=100)
plt.title("Distribución del precio de acuerdo a si vuelve o no en genero femenino")
sns.boxplot(
    data=df[(df.precio_ticket <= 30) & (df.genero == 'mujer')],
    y="precio_ticket",
    x="volveria",
)
plt.ylabel("Precio del ticket")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# %% [markdown]
#

# %%
plt.figure(dpi=100)
plt.title("Distribución del precio de acuerdo a si vuelve o no en genero masculino")
sns.boxplot(
    data=df[(df.precio_ticket <= 30) & (df.genero == 'hombre')],
    y="precio_ticket",
    x="volveria",
)
plt.ylabel("Precio del ticket")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# %% [markdown]
# ### Relación entre edad y genero

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de acuerdo a si vuelve o no en genero femenino")
sns.boxplot(
    data=df[(df.precio_ticket <= 30) & (df.genero == 'mujer')],
    y="edad",
    x="volveria",
)
plt.ylabel("Edad")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de acuerdo a si vuelve o no en genero masculino")
sns.boxplot(
    data=df[(df.precio_ticket <= 30) & (df.genero == 'hombre')],
    y="edad",
    x="volveria",
)
plt.ylabel("Edad")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# %% [markdown]
# ### Relacion entre tipo de sala y precio del ticket

# %%
plt.figure(dpi=100)
plt.title("Distribución del precio segun tipo de sala")
sns.boxplot(
    data=df[df.precio_ticket <= 30],
    y="precio_ticket",
    x="tipo_de_sala",
)
plt.ylabel("Precio del ticket")
plt.show()

# %%
plt.figure(dpi = 100)
plt.hist(df.precio_ticket[df.tipo_de_sala.str.strip() == '4d'], bins=10)
plt.title("Cantidad de encuestados segun precio ticket para sala 4d")
plt.show()

# %%
plt.figure(dpi = 100)
plt.hist(df.precio_ticket[df.tipo_de_sala.str.strip() == 'normal'], bins=50)
plt.title("Cantidad de encuestados segun precio ticket para sala normal")
plt.show()

# %%
df_mujeres = df[df.genero.str.strip() == 'mujer']
fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)
palermo = df_mujeres[df_mujeres["nombre_sede"] == "fiumark_palermo"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=palermo, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Sede")
axes[0].set_title("Palermo")


chacarita = df_mujeres[df_mujeres["nombre_sede"] == "fiumark_chacarita"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=chacarita, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Sede")
axes[1].set_title("Chacarita")

quilmes = df_mujeres[(df_mujeres["nombre_sede"] == "fiumark_quilmes") ] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=quilmes, ax=axes[2])
axes[2].set_ylabel("Cantidad")
axes[2].set_xlabel("Sede")
axes[2].set_title("Quilmes")
fig.suptitle("Cantidad de mujeres encuestadas segun tipo de sla y sede, y si volverian a ver Frozen 4")
plt.show()

# %% [markdown]
# Aumenta la probabilidad de volver en mujeres si la sala es normal o 3d. Incluso 4d en quilmes y chacarita

# %%
plt.figure(dpi=100)
sns.countplot(x="acompaniantes", hue="volveria", data=df)
plt.ylabel("Cantidad")
plt.xlabel("Acompañantes")
plt.title("Cantidad de encuestados segun cantidad de acompañantes y si volvería a ver Frozen 4")
plt.show()

# %%
df_hombres_volverian = df[(df.genero == 'hombre') & (df.volveria == 1)]

# %%
plt.figure(dpi=100)
sns.countplot(x="acompaniantes", data=df_hombres_volverian)
plt.ylabel("Cantidad")
plt.xlabel("Acompañantes")
plt.title("Cantidad de encuestados hombres que si volverian segun cantidad de acompañantes")
plt.show()


# %% [markdown]
# # Planteo del baseline

# %%
def baseline(fila):
    if fila['genero'] == 'hombre':
        return 0
    if fila['tipo_de_sala'] == '4d' and fila['nombre_sede'] == 'fiumark_palermo':
        return 0
    #if fila['parientes'] + fila['amigos'] > 3:
     #   return 0
    return 1


# %%
df['volveria_1'] = 0

# %%
df.head()

# %%

for nro_fila in range(len(df)):
    df.loc[nro_fila,'volveria_1'] = baseline(df.loc[nro_fila,:])


# %%
df.head()

# %%
df_aciertos = df[df.volveria == df.volveria_1]

# %%
len(df_aciertos[df_aciertos.genero == 'mujer']) / len(df[df.genero == 'mujer'])

# %%
len(df_aciertos[df_aciertos.genero == 'hombre']) / len(df[df.genero == 'hombre'])

# %%
len(df_aciertos) / len(df)

# %% [markdown]
# ### Distribucion de edad segun cantidad de amigos y parientes

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
figs.suptitle("Distribución de la edad segun cantidad de parientes en hombres")
sns.boxplot(
    data=df[(df.genero == 'mujer')],
    y="precio_ticket",
    x="volveria",
)
plt.ylabel("Precio del ticket")
plt.xticks([False, True], ["No", "Sí"])
plt.show()
