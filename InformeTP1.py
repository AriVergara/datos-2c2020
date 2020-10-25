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

# %%
import textwrap
def formatear_titulo(texto, longitud):
    return "\n".join(textwrap.wrap(texto, longitud))


# %%
def ajustar_leyenda(axes, labels_mapeados):
    for t in axes.get_legend().texts:
        t.set_text(labels_mapeados[t.get_text()])
        

def ajustar_leyenda_columna_volveria(axes, loc=None):
    labels_mapeados = {
        '0': "No",
        '1': "Sí"
    }
    ajustar_leyenda(axes, labels_mapeados)


# %% [markdown]
# ## Análisis general del dataset
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
# Al analizar los valores se puede obsevar que, si bien el `id_ticket` aparece repetido, el nombre de la persona que completó la encuesta es diferente, así como su `id_usuario`. Se procede a verificar que esto ocurre para todas las entradas.

# %%
print(f'Cantidad de filas con id_ticket y id_usuario repetido: {len(df[df.duplicated(subset=["id_ticket", "id_usuario"])])}')
print(f'Cantidad de filas con id_ticket y id_usuario repetido: {len(df[df.duplicated(subset=["id_ticket", "nombre"])])}')

# %% [markdown]
# A partir de esto, se decidió eliminar la columna del dataset, dado que no parece aportar información para determinar si una persona va volver al cine.

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
ajustar_leyenda_columna_volveria(axes[0])


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
# Se observa que el único tipo de sala que presenta una diferencia notoria es la de 4d (el 76% de los que van a esta sala optan por no volver), que a su vez es la que más entradas presenta en el dataset (aproximadamente un 55%). Considerando esto, que el tipo de sala sea 4d podría servir en el baseline para determinar si una persona vuelve.
#
# Se repite el análisis pero para la columna `nombre_sede`.

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
sns.countplot(x="nombre_sede", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Sede")
axes[0].set_title("Cantidad de encuestados segun sede del cine y si volvería a ver Frozen 4")
ajustar_leyenda_columna_volveria(axes[0])


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
ajustar_leyenda_columna_volveria(axes[0])


chacarita = df[df["nombre_sede"] == "fiumark_chacarita"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=chacarita, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Sede")
axes[1].set_title("Chacarita")
ajustar_leyenda_columna_volveria(axes[1])

quilmes = df[df["nombre_sede"] == "fiumark_quilmes"] 
sns.countplot(x="tipo_de_sala", hue="volveria", data=quilmes, ax=axes[2])
axes[2].set_ylabel("Cantidad")
axes[2].set_xlabel("Sede")
axes[2].set_title("Quilmes")
ajustar_leyenda_columna_volveria(axes[2])


# %% [markdown]
# Este gráfico no aporto ningún insight interesante.

# %% [markdown]
# ### Columna `genero`

# %% [markdown]
# Se aplica el mismo análisis inicial que para las columnas anteriores.

# %%
df.genero.value_counts()

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8))
sns.countplot(x="genero", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Género")
axes[0].set_title("Cantidad de encuestados segun género y si volvería a ver Frozen 4")
ajustar_leyenda_columna_volveria(axes[0])


sns.countplot(x="genero", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Género")
axes[1].set_title("Cantidad de encuestados segun género")
plt.show()

# %% [markdown]
# Se observa que la gran mayoría de los hombres opta por no volver, mientras que con las mujeres ocurre lo contrario. A su vez, hay mayor cantidad de hombres (64%) que mujeres encuestadas (36%). Las tendencias que se observan en esta columna aportan información valiosa a la hora de realizar la predicción. Si el baseline utilizace solamente esta columna, clasificando a los hombres como que no volverían y a las mujeres como que sí lo harían, se obtendria un accuracy aceptable con este mismo dataset.
#

# %% [markdown]
# ### Columna `edad`

# %%
df.edad.describe()

# %% [markdown]
# Se puede observar que las edades son valores de tipo float (el mínimo es 3.42). Inicialmente se esperaban que fueran de tipo entero, pero la edad no es un valor discreto, ya que se pueden contar tanto los meses como los días además de los años. A su vez, existen registros de encuestas en donde no se indica la edad, como se explicó anteriormente.
#
# Se procede a visualizar la distribución de valores que tiene esta columna:

# %%
df_edad = df.dropna()

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de los encuestados a lo largo del dataset")
plt.hist(x="edad", data=df_edad, bins=40)
plt.ylabel("Cantidad")
plt.xlabel("Edad")
plt.show()

# %% [markdown]
# Se puede ver que una gran cantidad del público encuestado se encuentra entre los 20 y 40 años aproximadamente, como se vió al comienzo de esta sección.

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de acuerdo a si vuelve o no")
sns.boxplot(
    data=df_edad,
    y="edad",
    x="volveria",
)
plt.ylabel("Edad")
plt.xticks([False, True], ["No", "Sí"])
plt.show()

# %% [markdown]
# Este gráfico muestra que la edad se distribuye de forma similar para ambas clases. Por si sóla no permite concluir algo respecto a si una persona volvería o no. 

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
# Una hipótesis lógica sería pensar que las personas que pagaron un precio alto por las entradas no querrían volver. Sin embargo, se observa que la mayoría de las que deciden no volver son las que pagaron un precio menor. Esto podría estar relacionado con que lo más importante a la hora de querer volver al cine para ver una secuela es si uno disfruto la película y no el precio de la entrada (esta información no se encuentra en el dataset).

# %% [markdown]
# ### Columnas `parientes` y `amigos`

# %% [markdown]
# Primero se analiza la columna `amigos`:

# %%
df.amigos.describe()

# %%
df.amigos.value_counts()

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(10, 5))

plt.subplots_adjust(wspace=0.4)
sns.countplot(x="amigos", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad de encuestados")
axes[0].set_xlabel("Cantidad de amigos")
axes[0].set_title("Cantidad de encuestados según el número \n de amigos que lo acompañó y si volvería")
ajustar_leyenda_columna_volveria(axes[0])


sns.countplot(x="amigos", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad de encuestados")
axes[1].set_xlabel("Cantidad de amigos")
axes[1].set_title("Cantidad de encuestados según el \n número de amigos que lo acompañó")



# %%
amigos_value_counts = df.amigos.value_counts()
cantidad_de_encuestados = len(df)
print(f'Porcentaje de encuestados que van sin ningún amigo: {amigos_value_counts[0] / cantidad_de_encuestados * 100}')
print(f'Porcentaje de encuestados que van a lo sumo con un amigo: {(amigos_value_counts[0] + amigos_value_counts[1]) / cantidad_de_encuestados * 100}')

# %% [markdown]
# El 69.5% de los encuestados fue sin ningún amigo y si se tiene en cuenta a lo sumo un amigo se alcanza el 91.8%. La cantidad de personas que va con solo un amigo presenta proporciones similares entre los que volverían y los que no.

# %% [markdown]
# Se repite el análisis para la columna `parientes`:

# %%
df.parientes.describe()

# %%
df.parientes.value_counts()

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(10, 5))

plt.subplots_adjust(wspace=0.4)
sns.countplot(x="parientes", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad de encuestados")
axes[0].set_xlabel("Cantidad de parientes")
axes[0].set_title("Cantidad de encuestados según el número \n de parientes que lo acompañó y si volvería")
ajustar_leyenda_columna_volveria(axes[0])


sns.countplot(x="parientes", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad de encuestados")
axes[1].set_xlabel("Cantidad de parientes")
axes[1].set_title("Cantidad de encuestados según el \n número de parientes que lo acompañó")

plt.show()

# %%
parientes_value_counts = df.parientes.value_counts()
print(f'Porcentaje de encuestados que van sin ningún amigo: {parientes_value_counts[0] / cantidad_de_encuestados * 100}')
print(f'Porcentaje de encuestados que van a lo sumo con un amigo: {(parientes_value_counts[0] + parientes_value_counts[1]) / cantidad_de_encuestados * 100}')

# %% [markdown]
# Ocurre algo similar a la columna `amigos`, dado que el 75.7% fue sin ningún familiar y el 89.5% fue a lo sumo con un pariente. En este caso, la cantidad de encuestados que van con dos familiares es más cercana a los que van con uno. 

# %% [markdown]
# Teniendo en cuenta estas cuestiones, se quiere analizar si los encuestados fueron acompañados o no, independientemente de si eran amigos o parientes. Para esto se agrega la columna `acompaniantes`, que consiste en sumar las columnas `parientes` y `amigos`.

# %%
df['acompaniantes'] = df.parientes + df.amigos 

# %%
acompaniantes_value_counts = df.acompaniantes.value_counts()
print(f'Porcentaje de encuestados que van sin acompañantes: {acompaniantes_value_counts[0] / cantidad_de_encuestados * 100}')
print(f'Porcentaje de encuestados que van a lo sumo con un acompañante: {(acompaniantes_value_counts[0] + acompaniantes_value_counts[1]) / cantidad_de_encuestados * 100}')
print(f'Porcentaje de encuestados que van a lo sumo con dos acompañantes: {(acompaniantes_value_counts[0] + acompaniantes_value_counts[1] + acompaniantes_value_counts[2]) / cantidad_de_encuestados * 100}')

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, dpi=100, figsize=(10, 5))

plt.subplots_adjust(wspace=0.4)
sns.countplot(x="acompaniantes", hue="volveria", data=df, ax=axes[0])
axes[0].set_ylabel("Cantidad de encuestados")
axes[0].set_xlabel("Cantidad de acompañantes")
axes[0].set_title("Cantidad de encuestados según el número \n de acompañantes y si volvería")
ajustar_leyenda_columna_volveria(axes[0])


sns.countplot(x="acompaniantes", data=df, ax=axes[1])
axes[1].set_ylabel("Cantidad de encuestados")
axes[1].set_xlabel("Cantidad de acompañantes")
axes[1].set_title("Cantidad de encuestados según el \n número de acompañantes")
plt.show()

# %% [markdown]
# Al combinar la cantidad de amigos y parientes en una sola columna, no se observa un comportamiento diferente. Se puede destacar que las proporciones para gente que va con uno y dos acompañantes son similares (ganando aquellos que volverían a ver Frozen 4). 
#
# Se esperaría que la mayor cantidad de encuestados vaya con al menos un acompañante, dado que no es tan común ver gente sola, y menos considerando que la encuesta es sobre una película animada como Frozen 3.

# %% [markdown]
# ## Relacionando columnas

# %% [markdown]
# A raíz del análisis individual de cada columna, surgieron las siguientes preguntas:
# - ¿Qué variables influyen en que los hombres decidan volver? ¿ Y cuáles provocan lo inverso en las mujeres?
# - ¿Es el tipo de sala importante para clasificar a las personas?
# - ¿La edad influye en la decisión de los encuestados? ¿Y la cantidad de acompañantes?

# %% [markdown]
# ### Relacionando Genero y Edad

# %% [markdown]
# Se buscó relacionar el genero con otras variables del dataset, con el objetivo de verificar otros factores importantes que puedan influenciar a ambos sexos a la hora de elegir si volverian o no a ver Frozen 4.
#
# Entre esas variables, se decidió comenzar con la edad:

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de los encuestados de acuerdo a su genero")
sns.boxplot(
    data=df,
    y="edad",
    x="genero",
)
plt.ylabel("Edad")
plt.xlabel("Género")
plt.show()

# %% [markdown]
# Tal como refleja el grafico, el rango etario de los encuestados es similar en ambos generos, ubicandose la mediana cerca de los 30 años de edad y el rango intercuartil entre los 20 y los 40 años. 

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad de los encuestados de acuerdo a su genero")
sns.boxplot(
    data=df,
    y="edad",
    x="genero",
    hue="volveria"
)
plt.ylabel("Edad")
plt.xlabel("Género")
plt.legend(loc="upper center")
ajustar_leyenda(plt.gcf().axes[0], {'0': 'No volveria', '1': 'Volveria'})
plt.show()

# %% [markdown]
# Se representan los mismos datos pero diferenciando la decisión sobre volver o no a ver Frozen 4. 

# %% [markdown]
# En principio, no se extraen resultados significativos de las visualizaciones realidadas, las mismas reflejan las desiciones de cada genero graficadas en [Columna genero](#Columna-genero)

# %% [markdown]
# ### Relacionando Genero y Acompañantes

# %% [markdown]
# En vista de que las columnas `parientes` y `amigos` tienen comportamientos similares tanto separadas como combinadas, se decide relacionar el genero de los encuestados con la cantidad total de acompañantes.

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
sns.countplot(x="acompaniantes", hue="volveria", data=df[df.genero == 'hombre'], ax=axes[0])
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Acompañantes")
axes[0].set_title('Hombres')
ajustar_leyenda_columna_volveria(axes[0])

sns.countplot(x='acompaniantes', hue='volveria', data=df[df.genero=='mujer'], ax=axes[1])
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Acompañantes")
axes[1].set_title('Mujeres')
ajustar_leyenda_columna_volveria(axes[1])
figs.suptitle("Cantidad de encuestados segun cantidad de acompañantes y si volveria a ver Frozen 4")
plt.show()

# %%
df_hombres = df[df.genero == 'hombre']
df_hombres_volveria = df_hombres[df_hombres.volveria == 1]
hombres_volverian_vc = df_hombres_volveria.acompaniantes.value_counts()

# %%
print(f'Porcentaje de hombres que fueron solos y declararon que volverian: {hombres_volverian_vc[0] / len(df_hombres[df_hombres.acompaniantes == 0]) * 100}')
print(f'Porcentaje de hombres que fueron con un acompañante y declararon que volverian: {hombres_volverian_vc[1] / len(df_hombres[df_hombres.acompaniantes == 1]) * 100}')
print(f'Porcentaje de hombres que fueron con dos acompañantes y declararon que volverian: {hombres_volverian_vc[2] / len(df_hombres[df_hombres.acompaniantes == 2]) * 100}')

# %% [markdown]
# Pueden extraerse dos observaciones del resultado de este analisis:
# - En el genero masculino, la mayor diferencia entre respuestas positivas y negativas sobre si volverian a ver Frozen 4 se encuentra cuando declaran haber ido sin acompañantes. Luego, al aumentar el numero de acompañantes, se reduce la diferencia.
# - En el genero femenino el comportamiento es similar, con la diferencia que el mayor porcentaje es de respuestas positivas.

# %% [markdown]
# Recordando lo visto en [el analisis de los acompañantes](#Columnas-parientes-y-amigos) se puede adjudicar este comportamiento a la diferencia en cantidad de encuestas respecto del numero de acompañantes.

# %% [markdown]
# ### Relacionando Genero y Precio del ticket

# %% [markdown]
# En busqueda de una relación con mayor fuerza entre el genero y otros factores, se buscó relacionarlo con el precio pagado por la entrada:

# %%
figs, axes = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=100)
plt.suptitle("Distribución del precio de acuerdo a si vuelve o no en ambos generos")
sns.boxplot(
    data=df[(df.precio_ticket <= 30) & (df.genero == 'mujer')],
    y="precio_ticket",
    x="volveria",
    ax=axes[0]

)
axes[0].set_ylabel("Precio del ticket")
axes[0].set_title("Mujeres")

sns.boxplot(
    data=df[(df.precio_ticket <= 30) & (df.genero == 'hombre')],
    y="precio_ticket",
    x="volveria",
    ax=axes[1]

)
axes[1].set_ylabel("Precio del ticket")
axes[1].set_title("Hombres")
plt.show()

# %% [markdown]
# # TODO revisar esto o sacarlo

# %% [markdown]
# ### Relacionando Genero, Tipo de Sala y Sede

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)
sedes = [sede.split('_')[1].capitalize() for sede in df_mujeres.nombre_sede.unique()]
def graficar_por_genero_y_sede(genero, axes):
    df_genero = df[df.genero == genero].dropna()
    ax = 0
    for sede in df_mujeres.nombre_sede.unique():
        df_sede = df_genero[df_genero.nombre_sede == sede]
        sns.countplot(x="tipo_de_sala", hue="volveria", data=df_sede, ax=axes[ax])
        axes[ax].set_ylabel("Cantidad")
        axes[ax].set_xlabel("Sede")
        axes[ax].set_title(sedes[ax])
        ajustar_leyenda_columna_volveria(axes[ax])
        ax += 1
graficar_por_genero_y_sede('mujer', axes)
fig.suptitle("Cantidad de mujeres encuestadas según tipo de sala y sede, y si volverian a ver Frozen 4")
plt.show()

# %% [markdown]
# Puede notarse en los gráficos que, en el género femenino, la respuesta negativa sobre si volvería a ver Frozen 4 aumenta cuando se trata de una sala 4d. Incluso supera a la respuesta positiva para la sede de Palermo específicamente.

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)
graficar_por_genero_y_sede('hombre', axes)
fig.suptitle("Cantidad de hombres encuestados según tipo de sala y sede, y si volverian a ver Frozen 4")
plt.show()

# %% [markdown]
# Por otro lado, no puede extraerse una conclusion certera haciendo el mismo analisis sobre el genero masculino

# %% [markdown]
# ### Relacionando tipo de sala y precio del ticket

# %% [markdown]
# En busca de ampliar la conclusión sobre el aumento de respuestas negativas en mujeres que fueron a salas 4d, se decidió relación el tipo de sala con el precio abonado:

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

# %% [markdown]
# Aqui se notó un resultado inesperado ya que, visto desde el sentido comun, se esperaba un rango de precio mas elevado para las salas 4d y 3d, respecto de la sala normal.

# %% [markdown]
# ### Relacionando Edad con Tipo de Sala

# %% [markdown]
# Se busca observar la influencia de la edad al elegir un tipo de sala. De este análisis podría extraerse una condición determinante a la hora de decidir si una persona volvería o no a ver Frozen 4.

# %%
plt.figure(dpi=100)
plt.title("Distribución de la edad según tipo de sala")
sns.boxplot(
    data=df,
    y="edad",
    x="tipo_de_sala",
)
plt.ylabel("Edad")
plt.xlabel("Tipo de Sala")
plt.show()

# %% [markdown]
# Se divide el grafico anterior por género en busca de una mejor observación:

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(6.4 * 2, 4.8), sharey=True)

sns.boxplot(data=df[df.genero == 'hombre'], y="edad", x="tipo_de_sala", ax=axes[0])
axes[0].set_ylabel("Edad")
axes[0].set_xlabel("Tipo de Sala")
axes[0].set_title("Hombres")
    
sns.boxplot(data=df[df.genero == 'mujer'], y="edad", x="tipo_de_sala", ax=axes[1])
axes[1].set_ylabel("Edad")
axes[1].set_xlabel("Tipo de Sala")
axes[1].set_title("Mujeres")

fig.suptitle("Distribucion de la edad segun el tipo de sala")
plt.show()

# %% [markdown]
# No puede extraerse una conclusión del grafico resultante. El comportamiento es similar para ambos géneros y no se refleja en él alguna característica que influencie en la decisión de volver o no a ver Frozen 4.

# %% [markdown]
# ### Relacionando la Edad con la cantidad de Acompañantes

# %% [markdown]
# Se representa a continuación una distibucion de la cantidad promedio de acompañantes segun la edad, a modo de observacion:

# %%
plt.figure(dpi=150)
sns.lineplot(
    data=df, x='edad', y='acompaniantes', hue='volveria', estimator='mean'
)
plt.show()

# %% [markdown]
# Si bien no es la mejor forma de graficarlo, se puede ver una relación entre la edad y los acompañantes respecto de si volveria (Agregue la condicion al baseline y suma un 1% creo. Pero tengo miedo que sea muy overfitting

# %% [markdown]
# # TODO ver esto de arriba

# %% [markdown]
# Pruebo lo mismo con precio tickey y edad

# %%
plt.figure(dpi=150)
sns.lineplot(
    data=df, x='edad', y='precio_ticket', hue='volveria', estimator='mean'
)
plt.show()

# %%
plt.figure(dpi=150)
sns.lineplot(
    data=df, x='edad', y='parientes', hue='volveria', estimator='mean'
)
plt.show()

# %%
plt.figure(dpi=150)
sns.lineplot(
    data=df, x='edad', y='amigos', hue='volveria', estimator='mean'
)
plt.show()


# %% [markdown]
# ## Planteo del baseline

# %%
def baseline(fila):
    if fila['edad'] < 20 and fila['parientes'] + fila['amigos'] <= 3:
        return 1
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

# %%
