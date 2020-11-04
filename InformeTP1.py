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

# # TP Parte 1 - Organización de Datos 2C 2020

# #### Alumnos
# - Alejo Rodriguez, 99869
# - Ariel Vergara, 97010

# + jupyter={"source_hidden": true}
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.metrics import accuracy_score

# + jupyter={"source_hidden": true}
sns.set()


# + jupyter={"source_hidden": true}
def ajustar_leyenda(axes, labels_mapeados, label_leyenda, loc=None):
    if loc:
        axes.legend(title=label_leyenda, loc=loc)
    else:
        axes.legend(title=label_leyenda)
    for t in axes.get_legend().texts:
        t.set_text(labels_mapeados[t.get_text()])
        

def ajustar_leyenda_columna_volveria(axes, loc=None):
    labels_mapeados = {
        '0': "No",
        '1': "Sí"
    }
    ajustar_leyenda(axes, labels_mapeados, "Volvería", loc)


# -

# Se proporcionaron dos datasets para realizar el trabajo práctico.

# El primero contiene dos columnas, el `id_usuario`y el booleano `volveria`, siendo esta última la variable target. 

# + jupyter={"source_hidden": true}
df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_volvera.head()
# -

# El segundo dataset contiene el resto de los datos encuestados.

# + jupyter={"source_hidden": true}
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos.head()

# + jupyter={"source_hidden": true}
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

# + jupyter={"source_hidden": true}
df.head()
# -

# A partir de este nuevo dataset se procederá con el análisis, algunas preguntas interesntes a tratar de responder son:
# - ¿Predomina el público infantil?
# - ¿Las personas que deciden volver fueron acompañadas?
# - ¿Predomina algún género sobre el otro en los encuestados que quisieran volver a ver Frozen 4?
# - ¿Es importante el precio a la hora de decidir si volver a ver Frozen 4?

# ## Análisis general del dataset
#
# Se cuenta con dos datasets diferentes, los cuales se pueden vincular mediante la columna `id_usuario`. 

# + jupyter={"source_hidden": true}
from pandas_profiling import ProfileReport

report = ProfileReport(
    df, title='Resumen de datos TP Parte 1', explorative=True, lazy=False
)

# + jupyter={"source_hidden": true}
#report.to_file('reporte.html')
report.to_widgets()
# -

# TODO: Se debe agarrar cada una de las afirmaciones y pasarlas a un gráfico.
#
# TODO GENERAL: Pasar al esquema "pregunta, analisis, respuesta"

# Se generó un reporte de Pandas Profiling para tener una visión general del dataset. Se observaron algunas cuestiones interesantes:
#
# - La clase mayoritaria en el dataset es la que corresponde a las personas que no volverían a ver Frozen 4 (61,6% de los encuestados).
# - Las únicas columnas que presentan missing values son `nombre_sede`, `fila` y `edad`, con un 0.25% (solo dos entradas del dataset), 20% y 77.9% de valores nulos respectivamente.
# - Existen dos variables categóricas ordinales, que son las columnas `amigos` y `parientes`.
# - Hay mayor cantidad de encuestados hombres que mujeres.
# - La mayor cantidad de las encuestas corresponden a la sede de Palermo.
# - La sala 4d es la más elegida por los encuestados.
# - La mayoría de los encuestados fue a ver Frozen 3 sin ningún pariente y/o amigo.
#
# Dado que la columna `fila` presenta una gran cantidad de missing values, se optó por eliminarla del dataframe.

# + jupyter={"source_hidden": true}
df.drop(axis=1, columns=["fila"], inplace=True)
# -

# A su vez, todos los valores de la columna `nombre` son únicos, y al ser el nombre del encuestado no aporta información respecto a su decisión sobre volver a ver Frozen 4. Por ello, se decidió eliminar la columna del dataset para el análisis.

# + jupyter={"source_hidden": true}
df.drop(axis=1, columns=["nombre"], inplace=True)
# -

# También se observó que la columna `id_ticket` presenta entradas repetidas, pero considerando que los valores de `id_usuario` y `nombre` son únicos, estas repeticiones no aportan información para la decisión (el que más se repite lo hace 7 veces). Por ello se decidió eliminar la columna.

# + jupyter={"source_hidden": true}
df.drop(axis=1, columns=["id_ticket"], inplace=True)
# -

# A continuación se analizan las demás columnas presentes en el dataset en relación con la columna objetivo.

# ### Columnas `tipo_de_sala` y `nombre_sede`

# Primero, se analiza la cantidad de encuestados considerando el tipo de sala y si volvería o no a ver Frozen 4.

# TODO: como se distribuyen los encuestados segun tipo de sala

# + jupyter={"source_hidden": true}
salas = df.tipo_de_sala.unique()

fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
fig.suptitle("Análisis de encuestados según el tipo de sala y si volvería a ver Frozen 4")

sns.countplot(x="tipo_de_sala", hue="volveria", data=df, ax=axes[0], order=salas)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Tipo de sala")
axes[0].set_title("Cantidad para cada sala")
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df.groupby("tipo_de_sala")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='tipo_de_sala',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
    order=salas
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Tipo de sala")
axes[1].set_title("Porcentaje para cada sala")
ajustar_leyenda_columna_volveria(axes[1])

plt.show()

# + jupyter={"source_hidden": true}
print("Cantidad de encuestados por tipo de sala")
display(df.tipo_de_sala.value_counts())
print(" ")
print("Porcentaje de encuestados por tipo de sala")
display(df.tipo_de_sala.value_counts().div(df.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
cuatro_d_value_counts = df[df.tipo_de_sala == '4d'].volveria.value_counts()
cantidad_de_encuestados = len(df)
print(f'Porcentaje de encuestados que fueron a 4d y no volverían : { cuatro_d_value_counts[0] / (cuatro_d_value_counts[0] + cuatro_d_value_counts[1]) * 100}')
# -

#
# Se observa que el único tipo de sala que presenta una diferencia notoria es la de 4d (el 76% de los que van a esta sala optan por no volver), que a su vez es la que más entradas presenta en el dataset (aproximadamente un 55%, como se vió en el reporte). Considerando esto, que el tipo de sala sea 4d podría servir en el baseline para determinar si una persona vuelve.
#
# Se repite el análisis pero para la columna `nombre_sede`.

# + jupyter={"source_hidden": true}
sedes = df.nombre_sede.dropna().unique()

fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
fig.suptitle("Análisis de encuestados según la sede y si volvería a ver Frozen 4")

sns.countplot(x="nombre_sede", hue="volveria", data=df, ax=axes[0], order=sedes)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Sede")
axes[0].set_title("Cantidad para cada sede")
ajustar_leyenda_columna_volveria(axes[0])

sns.barplot(
    data=df.groupby("nombre_sede")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='nombre_sede',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
    order=sedes
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Sede")
axes[1].set_title("Porcentaje para cada sede")
ajustar_leyenda_columna_volveria(axes[1])

plt.show()

# + jupyter={"source_hidden": true}
print("Cantidad de encuestados por sede")
display(df.nombre_sede.value_counts())
print(" ")
print("Porcentaje de encuestados por sede")
display(df.nombre_sede.value_counts().div(df.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
cuatro_d_value_counts = df[df.nombre_sede == 'fiumark_palermo'].volveria.value_counts()
cantidad_de_encuestados = len(df)
print(f'Porcentaje de encuestados que fueron a Palermo y no volverían : { cuatro_d_value_counts[0] / (cuatro_d_value_counts[0] + cuatro_d_value_counts[1]) * 100}')
# -

# Como se vió en el reporte, la mayoría de las encuestas correspondern a Palermo (aproximadamente un 73%) y esta es la única sede donde se observa una diferencia notoria entre ambas clases. Esta se debe seguramente a que hay más encuestados de una clase que de otra, como se vio al comienzo del análisis, dado que la proporción es similar (66% de las personas no volverían), por lo que pareciera ser que no es algo que se pueda asociar a la sede de Palermo.
#
# Se optó por ver si existía alguna diferencia entre los tipos de sala para cada una de las sedes:

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=2, ncols=3, dpi=100, figsize=(6.4 * 3, 4.8 * 2 + 2), sharey='row')
salas = df.tipo_de_sala.unique()
fig.tight_layout(pad=5)
#Palermo
palermo = df[df["nombre_sede"] == "fiumark_palermo"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=palermo, ax=axes[0][0], order=salas)
axes[0][0].set_ylabel("Cantidad")
axes[0][0].set_xlabel("Tipo de sala")
axes[0][0].set_title("Palermo - Cantidad para cada sala")
ajustar_leyenda_columna_volveria(axes[0][0])

sns.barplot(
    data=palermo.groupby("tipo_de_sala")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='tipo_de_sala',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1][0],
    order=salas
)
axes[1][0].set_ylabel("Porcentaje (%)")
axes[1][0].set_xlabel("Tipo de sala")
axes[1][0].set_title("Palermo - Porcentaje para cada sala")
ajustar_leyenda_columna_volveria(axes[1][0])


#Chacarita
chacarita = df[df["nombre_sede"] == "fiumark_chacarita"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=chacarita, ax=axes[0][1], order=salas)
axes[0][1].set_ylabel("Cantidad")
axes[0][1].set_xlabel("Tipo de sala")
axes[0][1].set_title("Chacarita - Cantidad para cada sala")
ajustar_leyenda_columna_volveria(axes[0][1])

sns.barplot(
    data=chacarita.groupby("tipo_de_sala")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='tipo_de_sala',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1][1],
    order=salas
)
axes[1][1].set_ylabel("Porcentaje (%)")
axes[1][1].set_xlabel("Tipo de sala")
axes[1][1].set_title("Chacarita - Porcentaje para cada sala")
ajustar_leyenda_columna_volveria(axes[1][1])


#Quilmes
quilmes = df[df["nombre_sede"] == "fiumark_quilmes"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=quilmes, ax=axes[0][2], order=salas)
axes[0][2].set_ylabel("Cantidad")
axes[0][2].set_xlabel("Tipo de sala")
axes[0][2].set_title("Quilmes - Cantidad para cada sala")
ajustar_leyenda_columna_volveria(axes[0][2])

sns.barplot(
    data=quilmes.groupby("tipo_de_sala")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='tipo_de_sala',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1][2],
    order=salas
)
axes[1][2].set_ylabel("Porcentaje (%)")
axes[1][2].set_xlabel("Tipo de sala")
axes[1][2].set_title("Quilmes - Porcentaje para cada sala")
fig.suptitle("Análisis de encuestados para cada sede por tipo de sala y si volvería a ver Frozen 4")
ajustar_leyenda_columna_volveria(axes[1][2])
# -


# En la sede de Quilmes, practicamente todos los encuestados corresponde a personas que fueron a salas 4d. Este gráfico no aportó ningún insight interesante, dado que ya se había podido oberver que en las salas 3d las proporciones entre las clases son similares mientras que en las normales hay mayor tendencia a querer volver a ver Frozen 4.

# ### Columna `genero`

# Se aplica el mismo análisis inicial que para las columnas anteriores.

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
generos = df.genero.unique()
fig.suptitle("Análisis de encuestados según el género y si volvería a ver Frozen 4")

sns.countplot(x="genero", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Género")
axes[0].set_title("Cantidad para cada género")
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df.groupby("genero")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='genero',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
    order=generos
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Género")
axes[1].set_title("Porcentaje para cada género")
ajustar_leyenda_columna_volveria(axes[1])
plt.show()

# + jupyter={"source_hidden": true}
print("Cantidad de encuestados por género")
display(df.genero.value_counts())
print(" ")
print("Porcentaje de encuestados por género")
display(df.genero.value_counts().div(df.pipe(len)).mul(100))
# -

# Se observa que la gran mayoría de los hombres opta por no volver, mientras que con las mujeres ocurre lo contrario. A su vez, como se vió en el Pandas Profiling, hay mayor cantidad de hombres (64%) que mujeres encuestadas (36%). Las tendencias que se observan en esta columna aportan información valiosa a la hora de realizar la predicción. Si el baseline utilizace solamente esta columna, clasificando a los hombres como que no volverían y a las mujeres como que sí lo harían, se obtendria un accuracy aceptable con este mismo dataset.
#

# ### Columna `edad`

# + jupyter={"source_hidden": true}
df.edad.describe()
# -

# Las edades son valores de tipo float (el mínimo es 3.42). Inicialmente se esperaban que fueran de tipo entero, pero la edad no es un valor discreto, ya que se pueden contar tanto los meses como los días además de los años. A su vez, existen registros de encuestas en donde no se indica la edad, como se explicó anteriormente.
#
# Se procede a visualizar la distribución de valores que tiene esta columna:

# + jupyter={"source_hidden": true}
df_edad = df.dropna(subset=["edad"])

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)

fig.suptitle("Distribución de la edad de los encuestados a lo largo del dataset")
axes[0].hist(x="edad", data=df_edad[df_edad.volveria == 1], bins=20)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Edad")
axes[0].set_title("Encuestados que volverían a ver Frozen 4")

axes[1].hist(x="edad", data=df_edad[df_edad.volveria == 0], bins=20)
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Edad")
axes[1].set_title("Encuestados que no volverían a ver Frozen 4")

plt.show()
# -

# Se puede ver que una gran cantidad del público encuestado se encuentra entre los 20 y 40 años aproximadamente, como se vió al comienzo de esta sección. Por lo que el público infantil no es el predominante, como se creyó inicialmente.
#
# Considerando que hay mas encuestados que no volverían, podemos asumir que las distribuciones son similares. Esto se puede corroborar en el siguiente grafico:

# + jupyter={"source_hidden": true}
plt.figure(dpi=100)
plt.title("Distribución de la edad de acuerdo a si vuelve o no")
sns.boxplot(
    data=df_edad,
    y="edad",
    x="volveria",
)
plt.ylabel("Edad")
plt.xlabel("Volvería")
plt.xticks([False, True], ["No", "Sí"])
plt.show()
# -

# Este gráfico confirma que la edad se distribuye de forma similar para ambas clases. Por si sóla no permite concluir algo respecto a si una persona volvería o no. 

# ### Columna `precio_ticket`

# De lo observado en el reporte de Pandas Profiling, se pudo definir que prácticamente todas las entradas tiene un valor menor o igual a 10 (el precio es un numero entero con valor mínimo 1 y máximo 50). Se analiza la distribución de los precios (de aquellas entradas con precio de ticket menor o igual a 10), de acuerdo a si vuelve o no.

# + jupyter={"source_hidden": true}
print(f'Porcentaje de tickets con precio menor o igual a 10: {len(df[df.precio_ticket <= 10]) / len(df) * 100}')

# + jupyter={"source_hidden": true}
plt.figure(dpi=100)
plt.title("Distribución del precio de acuerdo a si vuelve o no")
sns.boxplot(
    data=df[df.precio_ticket <= 10],
    y="precio_ticket",
    x="volveria",
)
plt.ylabel("Precio del ticket")
plt.xlabel("Volvería")
plt.xticks([False, True], ["No", "Sí"])
plt.show()
# -

# TODO: Analizar distribución de precios para ambas categorias con un histograma. Que pasa si nos quedamos con los precios 1 y 2? Tienen gran cantidad de Nos?

# Una hipótesis lógica sería pensar que las personas que pagaron un precio alto por las entradas no querrían volver. Sin embargo, se observa que la mayoría de las que deciden no volver son las que pagaron un precio menor. Esto podría estar relacionado con que lo más importante a la hora de querer volver al cine para ver una secuela es si uno disfruto la película y no el precio de la entrada (esta información no se encuentra en el dataset).
#
# El precio de la entrada no parece ser importante a la hora de clasificar a los encuestados.

# ### Columnas `parientes` y `amigos`

# Primero se analiza la columna `amigos`:

# + jupyter={"source_hidden": true}
df.amigos.value_counts()

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(6.4 * 2, 4.8))
fig.suptitle("Análisis de encuestados según la cantidad de amigos y si volvería a ver Frozen 4")

sns.countplot(x="amigos", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad de encuestados")
axes[0].set_xlabel("Cantidad de amigos")
axes[0].set_title("Cantidad de encuestados para cada cantidad de amigos")
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df.groupby("amigos")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='amigos',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Cantidad de amigos")
axes[1].set_title("Porcentaje para cada cantidad de amigos")
ajustar_leyenda_columna_volveria(axes[1])

plt.show()

# + jupyter={"source_hidden": true}
print("Cantidad de encuestados por cantidad de amigos")
display(df.amigos.value_counts())
print(" ")
print("Porcentaje de encuestados por cantidad de amigos")
display(df.amigos.value_counts().div(df.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
amigos_value_counts = df.amigos.value_counts()
cantidad_de_encuestados = len(df)
print(f'Porcentaje de encuestados que van a lo sumo con un amigo: {(amigos_value_counts[0] + amigos_value_counts[1]) / cantidad_de_encuestados * 100}')
# -

# El 69.5% de los encuestados fue sin ningún amigo y si se tiene en cuenta a lo sumo un amigo se alcanza el 91.8%. La cantidad de personas que va con solo un amigo presenta proporciones similares entre los que volverían y los que no.

# Se repite el análisis para la columna `parientes`:

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(6.4 * 2, 4.8))
fig.suptitle("Análisis de encuestados según la cantidad de parientes y si volvería a ver Frozen 4")

plt.subplots_adjust(wspace=0.4)
sns.countplot(x="parientes", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad de encuestados")
axes[0].set_xlabel("Cantidad de parientes")
axes[0].set_title("Cantidad de encuestados para cada cantidad de parientes")
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df.groupby("parientes")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='parientes',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Cantidad de parientes")
axes[1].set_title("Porcentaje para cada cantidad de parientes")
ajustar_leyenda_columna_volveria(axes[1])

plt.show()

# + jupyter={"source_hidden": true}
print("Cantidad de encuestados por cantidad de parientes")
display(df.parientes.value_counts())
print(" ")
print("Porcentaje de encuestados por cantidad de parientes")
display(df.parientes.value_counts().div(df.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
parientes_value_counts = df.parientes.value_counts()
print(f'Porcentaje de encuestados que van a lo sumo con un pariente: {(parientes_value_counts[0] + parientes_value_counts[1]) / cantidad_de_encuestados * 100}')
# -

# Ocurre algo similar a la columna `amigos`, dado que el 75.7% fue sin ningún familiar y el 89.5% fue a lo sumo con un pariente. En este caso, la cantidad de encuestados que van con dos familiares es más cercana a los que van con uno. 

# Teniendo en cuenta estas cuestiones, se quiere analizar si los encuestados fueron acompañados o no, independientemente de si eran amigos o parientes. Para esto se agrega la columna `acompaniantes`, que consiste en sumar las columnas `parientes` y `amigos`.

# + jupyter={"source_hidden": true}
df['acompaniantes'] = df.parientes + df.amigos 

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(6.4 * 2, 4.8))
fig.suptitle("Análisis de encuestados según la cantidad de acompañantes y si volvería a ver Frozen 4")

plt.subplots_adjust(wspace=0.4)
sns.countplot(x="acompaniantes", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad de encuestados")
axes[0].set_xlabel("Cantidad de acompañantes")
axes[0].set_title("Cantidad de encuestados para cada cantidad de acompañantes")
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df.groupby("acompaniantes")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='acompaniantes',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Cantidad de acompañantes")
axes[1].set_title("Porcentaje para cada cantidad de acompañantes")
ajustar_leyenda_columna_volveria(axes[1])

plt.show()

# + jupyter={"source_hidden": true}
print("Cantidad de encuestados por cantidad de acompaniantes")
display(df.acompaniantes.value_counts())
print(" ")
print("Porcentaje de encuestados por cantidad de acompaniantes")
display(df.acompaniantes.value_counts().div(df.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
acompaniantes_value_counts = df.acompaniantes.value_counts()
print(f'Porcentaje de encuestados que van a lo sumo con un acompañante: {(acompaniantes_value_counts[0] + acompaniantes_value_counts[1]) / cantidad_de_encuestados * 100}')
print(f'Porcentaje de encuestados que van a lo sumo con dos acompañantes: {(acompaniantes_value_counts[0] + acompaniantes_value_counts[1] + acompaniantes_value_counts[2]) / cantidad_de_encuestados * 100}')
# -

# Al combinar la cantidad de amigos y parientes en una sola columna, no se observa un comportamiento diferente. Se puede destacar que para los encuestados que fueorn con entre uno y tres acompañantes, el porcentaje de aqullos que deciden volver es mayor. Sin embargo, estos representan sólo el 30% de la cantidad de encuestados en total. 
#
# Se esperaría que la mayor cantidad de encuestados vaya con al menos un acompañante, dado que no es tan común ver gente sola en el cine, y menos considerando que la encuesta es sobre una película animada como Frozen 3.

# ## Relacionando columnas

# A raíz del análisis individual de cada columna, surgieron las siguientes preguntas:
# - ¿Qué variables influyen en que los hombres decidan volver? ¿ Y cuáles provocan lo inverso en las mujeres?
# - ¿Es el tipo de sala importante para clasificar a las personas?
# - ¿La edad influye en la decisión de los encuestados? ¿Y la cantidad de acompañantes?

# ### Relacionando `genero` y `edad`

# Se buscó relacionar el genero con otras variables del dataset, con el objetivo de verificar otros factores importantes que puedan influenciar a ambos sexos a la hora de elegir si volverían o no a ver Frozen 4.
#
# Entre esas variables, se decidió comenzar con la edad:

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)
df_hombres = df_edad[df_edad.genero == 'hombre']
fig.suptitle("Distribución de la edad en el género masculino")

#Hombres
axes[0].hist(x="edad", data=df_hombres[df_hombres.volveria == 1], bins=20)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Edad")
axes[0].set_title("Encuestados que volverían a ver Frozen 4")

axes[1].hist(x="edad", data=df_hombres[df_hombres.volveria == 0], bins=20)
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Edad")
axes[1].set_title("Encuestados que no volverían a ver Frozen 4")

plt.show()

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)
df_mujeres = df_edad[df_edad.genero == 'mujer']
fig.suptitle("Distribución de la edad en el género femenino")

#Hombres
axes[0].hist(x="edad", data=df_mujeres[df_mujeres.volveria == 1], bins=20)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Edad")
axes[0].set_title("Encuestadas que volverían a ver Frozen 4")

axes[1].hist(x="edad", data=df_mujeres[df_mujeres.volveria == 0], bins=20)
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Edad")
axes[1].set_title("Encuestadas que no volverían a ver Frozen 4")
plt.ylim(0,25)
plt.show()

# + jupyter={"source_hidden": true}
plt.figure(figsize=(6.4, 4.8), dpi=100)
plt.title("Distribución de la edad de los encuestados de acuerdo a su genero")
sns.boxplot(
    data=df,
    y="edad",
    x="genero",
    hue='volveria'
)
ajustar_leyenda_columna_volveria(plt.gcf().axes[0], 'upper center')
plt.ylabel("Edad")
plt.xlabel("Género")
plt.show()
# -

# Tal como refleja en los gráficos, el rango etario de los encuestados es similar en ambos géneros, ubicándose la mediana cerca de los 30 años de edad y el rango intercuartil entre los 20 y los 40 años. 

# En principio, no se extraen resultados significativos de las visualizaciones realidadas, las mismas reflejan las desiciones de cada genero graficadas en [Columna genero](#Columna-genero).

# ### Relacionando `genero` y `acompaniantes`

# En vista de que las columnas `parientes` y `amigos` tienen comportamientos similares tanto separadas como combinadas, se decide relacionar el género de los encuestados con la cantidad total de acompañantes.

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
acompaniantes = df.acompaniantes.unique().sort()
fig.suptitle("Análisis de encuestados de género masculino según cantidad de acompañantes")

sns.countplot(x="acompaniantes", hue="volveria", data=df[df.genero == 'hombre'], ax=axes[0], order=acompaniantes)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Acompañantes")
axes[0].set_title('Cantidad para cada número de acompañantes')
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df[df.genero == 'hombre'].groupby("acompaniantes")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='acompaniantes',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
    order=acompaniantes
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Acompañantes")
axes[1].set_title("Porcentaje para cada número de acompañantes")
ajustar_leyenda_columna_volveria(axes[1])
plt.show()

# + jupyter={"source_hidden": true}
df_hombres = df[df.genero == 'hombre']
print("Cantidad de encuestados hombres por cantidad de acompañantes")
display(df_hombres.acompaniantes.value_counts())
print(" ")
print("Porcentaje de encuestados hombres por cantidad de acompañantes")
display(df_hombres.acompaniantes.value_counts().div(df_hombres.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
acompaniantes = df.acompaniantes.unique().sort()
fig.suptitle("Análisis de encuestados de género femenino según cantidad de acompañantes")

sns.countplot(x="acompaniantes", hue="volveria", data=df[df.genero == 'mujer'], ax=axes[0], order=acompaniantes)
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Acompañantes")
axes[0].set_title('Cantidad para cada número de acompañantes')
ajustar_leyenda_columna_volveria(axes[0])


sns.barplot(
    data=df[df.genero == 'mujer'].groupby("acompaniantes")
    .volveria.value_counts(normalize=True)
    .rename("volveria_prop")
    .mul(100)
    .reset_index(),
    x='acompaniantes',
    y="volveria_prop",
    hue='volveria',
    ax=axes[1],
    order=acompaniantes
)
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Acompañantes")
axes[1].set_title("Porcentaje para cada número de acompañantes")
ajustar_leyenda_columna_volveria(axes[1])
plt.show()

# + jupyter={"source_hidden": true}
df_mujeres = df[df.genero == 'mujer']
print("Cantidad de encuestadas mujeres por cantidad de acompañantes")
display(df_mujeres.acompaniantes.value_counts())
print(" ")
print("Porcentaje de encuestadas mujeres por cantidad de acompañantes")
display(df_mujeres.acompaniantes.value_counts().div(df_mujeres.pipe(len)).mul(100))
# -

# Pueden extraerse dos observaciones del resultado de este analisis:
# - En el género masculino, la mayor diferencia entre respuestas positivas y negativas sobre si volverían a ver Frozen 4 se encuentra cuando declaran haber ido sin acompañantes. Luego, al aumentar el número de acompañantes, se reduce la diferencia. Sin embargo, casi el 97% de los encuestados hombres se encuentran entre los que fueron con, a lo sumo, 2 acompañantes. Ademas, el 73% de los hombres fueron a ver Frozen 3 sin acompañantes.
# - En el género femenino se mantiene constante el porcentaje de respuestas positivas, dandose el quiebre a partir de 4 acompañantes. Sin embargo, al igual que en los hombres, la cantidad de encuestados con un número de acompañantes mayor a 3 es muy pequeña. El 87% de las encuestadas fueron a ver Frozen 3 con a lo sumo 3 acompañantes.

# TODO: Aclarar que no es significativamente importante en los hombres

# Recordando lo visto en [el análisis de los acompañantes](#Columnas-parientes-y-amigos) se puede adjudicar este comportamiento a la diferencia en cantidad de encuestas respecto del número de acompañantes.


# ### Relacionando `genero`, `tipo_de_sala` y `nombre_sede`

# Primero se analizan las tendencias para cada género de acuerdo al tipo de sala:

# + jupyter={"source_hidden": true}
def graficar_por_genero(df, titulo):
    fig, axes = plt.subplots(nrows=2, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8 * 2), sharey='row')
    fig.tight_layout(pad=5)
    salas = df.tipo_de_sala.unique()
    ax = 0
    fig.suptitle(titulo)
    for genero in df.genero.unique():
        sns.countplot(x="tipo_de_sala", hue="volveria", data=df[df.genero == genero], ax=axes[0][ax], order=salas)
        axes[0][ax].set_ylabel("Cantidad")
        axes[0][ax].set_xlabel("Tipo de sala")
        axes[0][ax].set_title(f"{genero.capitalize()} - Cantidad para cada sala")
        ajustar_leyenda_columna_volveria(axes[0][ax])
        
        sns.barplot(
            data=df[df.genero == genero].groupby("tipo_de_sala")
            .volveria.value_counts(normalize=True)
            .rename("volveria_prop")
            .mul(100)
            .reset_index(),
            x='tipo_de_sala',
            y="volveria_prop",
            hue='volveria',
            ax=axes[1][ax],
            order=salas
        )
        axes[1][ax].set_ylabel("Porcentaje (%)")
        axes[1][ax].set_xlabel("Tipo de sala")
        axes[1][ax].set_title(f"{genero.capitalize()} - Porcentaje para cada sala")
        ajustar_leyenda_columna_volveria(axes[1][ax])

        ax += 1
    plt.show()


# + jupyter={"source_hidden": true}
graficar_por_genero(df, "Análisis de encuestados para cada género según tipo de sala y si volverian a ver Frozen 4")


# -

# Puede notarse en los gráficos que, en el género femenino, la proporción de respuesta negativa sobre si volvería a ver Frozen 4 aumenta cuando se trata de una sala 4d, incluso se equipara. Se procede a repetir este análisis pero diferenciando por sede:

# + jupyter={"source_hidden": true}
def graficar_por_genero_y_sede(df, genero, titulo):
    fig, axes = plt.subplots(nrows=2, ncols=3, dpi=150, figsize=(6.4 * 3, 4.8 * 2 + 2), sharey='row')
    fig.tight_layout(pad=5)
    salas = df.tipo_de_sala.unique()
    sedes = df.nombre_sede.dropna().unique()
    df_genero = df[df.genero == genero].dropna(subset=["nombre_sede"]) #Porque dos entradas no tienen sede
    ax = 0
    for sede in sedes:
        df_sede = df_genero[df_genero.nombre_sede == sede]
        sns.countplot(x="tipo_de_sala", hue="volveria", data=df_sede, ax=axes[0][ax], order=salas)
        axes[0][ax].set_ylabel("Cantidad")
        axes[0][ax].set_xlabel("Tipo de sala")
        axes[0][ax].set_title(f"{sede.split('_')[1].capitalize()} - Cantidad para cada sala")
        ajustar_leyenda_columna_volveria(axes[0][ax])
        
        
        sns.barplot(
            data=df_sede.groupby("tipo_de_sala")
            .volveria.value_counts(normalize=True)
            .rename("volveria_prop")
            .mul(100)
            .reset_index(),
            x='tipo_de_sala',
            y="volveria_prop",
            hue='volveria',
            ax=axes[1][ax],
            order=salas
        )
        axes[1][ax].set_ylabel("Porcentaje (%)")
        axes[1][ax].set_xlabel("Tipo de sala")
        axes[1][ax].set_title(f"{sede.split('_')[1].capitalize()} - Porcentaje para cada sala")
        ajustar_leyenda_columna_volveria(axes[1][ax])
        ax += 1
        
    fig.suptitle(titulo)
    plt.show()

# + jupyter={"source_hidden": true}
graficar_por_genero_y_sede(df,
                           'mujer', 
                           "Cantidad de mujeres encuestadas según tipo de sala y sede, y si volverian a ver Frozen 4")
# -

# Se observa como en la sede de Palermo hay una mayor proporción de mujeres que optan por no volver, mostrando un comportamiento diferente que en los demás tipo de salas e incluso sedes (además, en las otras sedes la cantidad de mujeres encuestadas es mucho menor).

# TODO: Poner las conclusiones junto con el grfico

# + jupyter={"source_hidden": true}
df_palermo_mujeres_4d = df[(df.nombre_sede == 'fiumark_palermo') & (df.genero == 'mujer') & (df.tipo_de_sala == '4d')]
print("Cantidad de encuestadas mujeres que fueron a Palermo a sala 4D")
display(df_palermo_mujeres_4d.volveria.value_counts())
print(" ")
print("Porcentaje de encuestadas mujeres que fueron a Palermo a sala 4D y volverían a ver Frozen 4")
display(df_palermo_mujeres_4d.volveria.value_counts().div(df_palermo_mujeres_4d.pipe(len)).mul(100))

# + jupyter={"source_hidden": true}
graficar_por_genero_y_sede(df,
                           'hombre', 
                           "Cantidad de hombres encuestados según tipo de sala y sede, y si volverian a ver Frozen 4")

# + jupyter={"source_hidden": true}
df_hombres_4d = df[(df.genero == 'hombre') & (df.tipo_de_sala == 'normal')]
print("Porcentaje de encuestados hombres que fueron a sala normal y volverían a ver Frozen 4")
display(df_hombres_4d.volveria.value_counts().div(df_hombres_4d.pipe(len)).mul(100))
# -

# Una observación que se hizo en este análisis es que, para la sala normal, la proporción de hombres que volverían a ver Frozen 4 es mayor que para los demás tipos de sala. Sin embargo, la cantidad de encuestados que entran dentro de esta categoría es baja, por lo tanto no aporta una condición sólida para determinar si un hombre volvería o no.

# ### Relacionando `tipo_de_sala` y `precio_ticket`

# En busca de ampliar la conclusión sobre el aumento de respuestas negativas en mujeres que fueron a salas 4d, se buscó relacionar el tipo de sala con el precio abonado:

# + jupyter={"source_hidden": true}
plt.figure(dpi=100)
plt.title("Distribución del precio según tipo de sala")
sns.boxplot(
    data=df[df.precio_ticket <= 30],
    y="precio_ticket",
    x="tipo_de_sala",
)
plt.ylabel("Precio del ticket")
plt.xlabel("Tipo de Sala")
plt.show()
# -

# Aqui se notó un resultado inesperado ya que, visto desde el sentido común, se esperaba un rango de precio más elevado para las salas 4d y 3d respecto de la sala normal.

# ### Relacionando `edad` con `tipo_de_sala`

# Se busca observar la influencia de la edad al elegir un tipo de sala de acuerdo al género.

# + jupyter={"source_hidden": true}
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)

sns.boxplot(data=df[df.genero == 'hombre'], y="edad", x="tipo_de_sala", ax=axes[0])
axes[0].set_ylabel("Edad")
axes[0].set_xlabel("Tipo de Sala")
axes[0].set_title("Hombres")
    
sns.boxplot(data=df[df.genero == 'mujer'], y="edad", x="tipo_de_sala", ax=axes[1])
axes[1].set_ylabel("Edad")
axes[1].set_xlabel("Tipo de Sala")
axes[1].set_title("Mujeres")

fig.suptitle("Distribución de la edad según el tipo de sala y género")
plt.show()
# -

# No puede extraerse una conclusión del gráfico resultante. El comportamiento es similar para ambos géneros y no se refleja en él alguna característica que influencie en la decisión de volver o no a ver Frozen 4.

# ### Relacionando la `edad` con `acompaniantes`

# Se analiza la cantidad de acompañantes con las que van los menores de edad:

# + jupyter={"source_hidden": true}
menores_de_edad = df[df.edad < 18]
plt.figure(dpi=150)
sns.countplot(x="acompaniantes", hue="volveria", data=menores_de_edad)
plt.ylabel("Cantidad")
plt.xlabel("Acompañantes")
plt.title("Cantidad de encuestados para cada número de acompañantes")
plt.show()

# + jupyter={"source_hidden": true}
print(f"Cantidad de encuestados menores de edad: {len(menores_de_edad)}")
print(f"Porcentaje de encuestados menores de edad: {len(menores_de_edad) / len(df) * 100}")


# -

# Si bien no es la mejor forma de graficarlo, se puede observar que son extremadamente pocos (4 encuestados) los menores que van solos al cine, lo cual se condice con lo que se ve normalmente cuando uno va a ver una película. A su vez, para una cantidad de acompañantes menor o igual a 3 la gran mayoría decide volver a ver Frozen 4, mientras que prácticamente todos los que van con 4 o más optan por lo contrario. 

# ## Armado del baseline

# A partir del análisis realizado, se concluyó que la columna más importante para definir si una persona volvería a ver Frozen 4 en el mismo cine es el género. Como se explicó anteriormente, aproximadamente el 80% de los hombres encuestados respondió que no volvería, mientras que el 70% de las mujeres dijo que sí lo haría. Si se clasificase a todos los hombres como que no volverían y a las mujeres como que sí volverían, se obtendría un accuracy superior al 75%. 
#
# A su vez, en el caso de las mujeres, se observó que para la sede de Palermo la mayoría de las que iba a la sala 4d optaba por no volver (65%). Para los demás tipos de sala y sede esto no ocurría, la gran mayoría elegía volver. Se puede aprovechar esta diferencia de comportamiento en salas 4d de la sede de Palermo para mejorar la clasificación de las mujeres encuestadas. Por ello, se decidió utilizar una moneda cargada para este caso en particular.
#
# Otra condición interesante encontrada fue que prácticamente todos los menores de edad que fueron al cine con 3 o menos acompañantes deciden volver, ocurriendo lo contrario con los que acudieron con más de 3 acompañantes. Sin embargo, el porcentaje de encuestados menores de edad es muy bajo (8%) y, como se dijo al comienzo del informe, el 20% de las entradas del dataset no tienen la edad cargada. Aún así, corriendo riesgo de un sobreajuste al dataset, se decidió utilizar esta condición en el baseline.
#
# Estas últimas dos condiciones tienen un peso mucho menor que la mencionada para el género, la mejora en la clasificación es muy pequeña en comparación a la obtenida al diferenciar entre hombres y mujeres.
#
# A partir de estas consideraciones se construyó el siguiente baseline:

def clasificar_encuestado(fila):
    if fila['edad'] < 18:
        acompaniantes = fila['parientes'] + fila['amigos']
        return 1 if acompaniantes <= 3 else 0
    if fila['genero'] == 'hombre':
        return 0
    if fila['tipo_de_sala'] == '4d' and fila['nombre_sede'] == 'fiumark_palermo':
        return np.random.binomial(1, 0.35) #El 35% de las mujeres que concurren a esta sala en esta sede volverian a ver Frozen 4
    return 1


# TODO: Cambiar moneda cargada porque no queremos un random dentro del codigo. Tener algo random dentro del baseline es un mal caso. Queremos que sea deterministico.

# + jupyter={"source_hidden": true}
def baseline(X):
    resultado = []
    for nro_fila in range(len(X)):
        resultado.append(clasificar_encuestado(df.loc[nro_fila,:]))
    return resultado


# -

# Se procede a verificar el accuracy obtenido usando la funcion `accuracy_score`de la libreria `sklearn`.

# + jupyter={"source_hidden": true}
prediccion = baseline(df)
accuracy_score(df.volveria, prediccion)
# -


# Como se puede ver, el accuracy obtenido cumple con los requisitos pedidos. 
