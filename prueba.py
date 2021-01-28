# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import preprocesing as pp

# ### Carga de Datasets

df_volvera = pd.read_csv('https://drive.google.com/uc?export=download&id=1km-AEIMnWVGqMtK-W28n59hqS5Kufhd0')
df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df_datos = pd.read_csv('https://drive.google.com/uc?export=download&id=1i-KJ2lSvM7OQH0Yd59bX01VoZcq8Sglq')
df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')

# ### Procesado de titulos (no se si se les llama asi jaja)

# Creo que sirve si vamos a utilizar KNNImputer, ya que puede mejorar la clasificacion dentro de las mujeres al separar entre 'Señora' y 'Señorita' (Quizas no aporte mucho)

df = pp.procesar_titulos(df)
df.head()

# ### Borrado de columnas

df = pp.borrar_columna_fila(df)
df = pp.borrar_columna_nombre(df)
df = pp.borrar_columna_id_ticket(df)
df.head()

# ### KNNImputer para las edades con missing values

df['edad_knn'] = pp.knn_imputer(df)
df.head()

# ### OneHotEncoding para las variables categoricas

df = pp.one_hot_encoding(df, 'nombre_sede')
df = pp.one_hot_encoding(df, 'genero')
df = pp.one_hot_encoding(df, 'tipo_de_sala')
df.head()

# ### Redondeo de edades

df.drop(['edad'],axis=1, inplace=True)
df = pp.redondear_edades(df, 'edad_knn')
df.head()


