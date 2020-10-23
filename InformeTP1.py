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

# %%
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns

# %% [markdown]
# ## Análisis general
#
# Se cuenta con dos datasets diferentes, los cuales se pueden vincular mediante la columna `id_usuario`. 

# %% [markdown]
# El primer dataset contiene el `id_usuario`y el booleano `volveria`. 

# %%
df_volvera = pd.read_csv('tp-2020-2c-train-cols1.csv')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_volvera

# %% [markdown]
# El segundo dataset contiene el resto de los datos encuestados.

# %%
df_datos = pd.read_csv('tp-2020-2c-train-cols2.csv')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos

# %%
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

# %%
df.head()

# %%
#from pandas_profiling import ProfileReport

#report = ProfileReport(
    #df, title='Resumen de datos TP Parte 1', explorative=True, lazy=False
#)

# %%
#report.to_file('reporte.html')

# %% [markdown]
# Se generó un reporte de Pandas Profiling (reporte.html) para tener una visión general del dataset. Se observaron algunas cuestiones interesantes:
#
# - la clase mayoritaria en el dataset es la que corresponde a las personas que no volverían a ver Frozen 4 (61,6% de los encuestados).
# - las únicas columnas que presentan missing values son `fila` y `edad`, con un 20% y 78% de valores nulos respectivamente.
#
# Dado que la columna `fila` presenta una gran cantidad de missing values, se optó por eliminarla del dataframe.

# %%
print(f'Porcentaje de valores nulos para la columna "fila": {df.fila.isnull().sum()/len(df) * 100}')

# %%
df.drop(axis=1, columns=["fila"], inplace=True)

# %% [markdown]
# A continuación se analizan las distintas columnas presentes en el dataset.

# %% [markdown]
# ### Columna `id_ticket`

# %%
df.head()

# %% [markdown]
# Se puede ver que columna `id_ticket` es un string. Una cuestión interesante es ver si existen entradas repetidas para esta columna.

# %%
df_tickets_repetidos = df[df.duplicated(subset=["id_ticket"])]
print(f'Cantidad de filas con id_ticket repetido: {len(df_tickets_repetidos)}')
df_tickets_repetidos.head()

# %% [markdown]
# Sin embargo, al analizar estos valores se puede obsevar que si bien el `id_ticket` aparece repetido, el nombre de la persona que completó la encuesta es diferente, así como su `id_usuario`. Se procede a verificar que esto ocurre para todas las entradas.

# %%
print(f'Cantidad de filas con id_ticket y id_usuario repetido: {len(df[df.duplicated(subset=["id_ticket", "id_usuario"])])}')
print(f'Cantidad de filas con id_ticket y id_usuario repetido: {len(df[df.duplicated(subset=["id_ticket", "nombre"])])}')

# %% [markdown]
# A partir de esto, se decidió eliminar la columna del dataset, dado que no parece aportar información para determinar si una persona va volver al cine

# %%
df.drop(axis=1, columns=["id_ticket"], inplace=True)

# %% [markdown]
# ### Columnas `tipo_de_sala` y `nombre_sede`

# %% [markdown]
# Primero, se analiza la cantidad de encuestados considerando el tipo de sala y si volvería o no.

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

# %%
df[df.tipo_de_sala == '4d'].volveria.value_counts()

# %%
df.tipo_de_sala.value_counts()

# %% [markdown]
#
# Se observa que único tipo de sala que presenta una diferencia notoria es la de 4d (el 76% de los que van a esta sala optan por no volver), que a su vez es la que más entradas presenta en el dataset (aproximadamente un 55%). Considerando esto, que el tipo de sala sea 4d podría servir en el baseline para determinar si una persona vuelve.
#
# Se repite el análisis pero para la columna nombre_sede.

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

# %%
df.nombre_sede.value_counts()

# %%
df[df.nombre_sede == "fiumark_palermo"].volveria.value_counts()

# %% [markdown]
# Se observa que la gran mayoría de las encuestas corresponden a la sede de Palermo (aproximadamente un 73%) y esta es la única donde se observa una diferencia notoria entre ambas clases. Esta se debe seguramente a que hay más encuestados de una clase que de otra, como se vio al comienzo del análisis, dado que la proporción es similar (66% de las personas no volverían), por lo que pareciera ser que no es algo que se pueda asociar a la sede de Palermo.
#
# Se optó por ver si existía alguna diferencia entre los tipos de sala para cada una de las sedes:

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


# %% [markdown]
# Este gráfico no aporto ningún insight interesante.

# %% [markdown]
# ### Columna `genero`

# %% [markdown]
# Se aplica el mismo análisis inicial que para las columnas anteriores.

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
# Se observa que la gran mayoría de los hombres opta por no volver, mientras que con las mujeres ocurre lo contrario. A su vez, hay mayor cantidad de hombres (64%) que mujeres encuestadas (36%). Las tendencias que se observan en esta columna aportan información valiosa a la hora de realizar la predicción. Si el baseline utilizace solamente esta columna, clasificando a los hombres como que no volverían y a las mujeres como que sí lo harían, se obtendria un accuracy aceptable.
#

# %% [markdown]
# ### Columna `edad`

# %% [markdown]
# ### Columna `precio`

# %% [markdown]
# De lo observado en el reporte de Pandas Profiling, se pudo definir que prácticamente todas las entradas tiene un valor menor o igual a 10 (el precio es un numero entero con valor mínimo 1 y máximo 50). Se analiza la distribución de los precios (de aquellas entradas con precio de tiket menor o igual a 10), de acuerdo a si vuelve o no.

# %%
print(f'Porcentaje de tickets con precio menor o igual a 10: {len(df[df.precio_ticket <= 10]) / len(df) * 100}')

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
# Una hipótesis lógica sería pensar que las personas que pagaron un precio alto por las entradas no querrían volver. Sin embargo, se observa que la mayoría de las que deciden no volver son las que pagaron un precio menor. Esto podría estar relacionado con que lo más importante a la hora de querer volver al cine para ver una secuela es si uno disfruto la película y no el precio de la entrada.

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
    data=df,
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
# Primero vamos a probar crear una nueva variable `acompañantes` para buscar relaciones entre las distintas variables y la cantidad de acompañantes con los que vieron la pelicula, sin importar de que tipo sean.

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
