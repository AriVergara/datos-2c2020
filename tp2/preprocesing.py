import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder
)

# --------- FUNCIONES DE PREPROCESAMIENTO ------------

RANDOM_STATE = 117

TEST_SIZE = 0.2

def procesamiento_arboles(df):
    #Se indica que columnas tenian edad nula
    df['edad_nan'] = np.where(df['edad'].isnull(), 1, 0)
    df['mujer_4d_palermo'] = np.where((df['genero'] == 'mujer') & (df['tipo_de_sala'] == '4d') & (df['nombre_sede'] == 'fiumark_palermo'), 1, 0)
    for sede in ['fiumark_palermo', 'fiumark_quilmes', 'fiumark_chacarita']:
        for sala in ['normal', '3d', '4d']:
            df[sede+'_'+sala] = np.where((df['tipo_de_sala'] == sala) & (df['nombre_sede'] == sede), 1, 0)
    #Se procesa el titulo de cada encuestado para que lo utilice el 
    #knnimputer para calcular los missing values en la edad.
    ##df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
    #Se borran columnas que no aportan datos significativos
    borrar_columna(df, 'fila', True)
    borrar_columna(df, 'nombre', True)
    borrar_columna(df, 'id_ticket', True)
    df['edad_knn'] = knn_imputer(df)
    #Se encodean las columnas categoricas
    #df = one_hot_encoding(df, 'nombre_sede')
    df = one_hot_encoding(df, 'genero')
    #df = one_hot_encoding(df, 'tipo_de_sala')
    #Se dropea la edad con missing values y se redondean los valores de edad calculados
    df.drop(['edad'],axis=1, inplace=True)
    df.drop(['precio_ticket'],axis=1, inplace=True)
    df = redondear_edades(df, 'edad_knn')
    #Se dropean las columnas titulo e id_usuario que no son utiles
    df.drop(columns=['id_usuario', 'nombre_sede', 'tipo_de_sala'], inplace=True)
    return df

def procesamiento_rf_1(df):
    #Se indica que columnas tenian edad nula
    df['edad_nan'] = np.where(df['edad'].isnull(), 1, 0)
    df['mujer_4d_palermo'] = np.where((df['genero'] == 'mujer') & (df['tipo_de_sala'] == '4d') & (df['nombre_sede'] == 'fiumark_palermo'), 1, 0)
    for sede in ['fiumark_palermo', 'fiumark_quilmes', 'fiumark_chacarita']:
        for sala in ['normal', '3d', '4d']:
            df[sede+'_'+sala] = np.where((df['tipo_de_sala'] == sala) & (df['nombre_sede'] == sede), 1, 0)
    #Se procesa el titulo de cada encuestado para que lo utilice el 
    #knnimputer para calcular los missing values en la edad.
    ##df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
    #Se borran columnas que no aportan datos significativos
    borrar_columna(df, 'fila', True)
    borrar_columna(df, 'nombre', True)
    borrar_columna(df, 'id_ticket', True)
    df['edad_knn'] = knn_imputer(df)
    #Se encodean las columnas categoricas
    #df = one_hot_encoding(df, 'nombre_sede')
    df = one_hot_encoding(df, 'genero')
    #df = one_hot_encoding(df, 'tipo_de_sala')
    #Se dropea la edad con missing values y se redondean los valores de edad calculados
    df.drop(['edad'],axis=1, inplace=True)
    df.drop(['precio_ticket'],axis=1, inplace=True)
    df = redondear_edades(df, 'edad_knn')
    #Se dropean las columnas titulo e id_usuario que no son utiles
    df.drop(columns=['id_usuario', 'nombre_sede', 'tipo_de_sala'], inplace=True)
    return df

def procesamiento_rf_2(df):
    #Se indica que columnas tenian edad nula
    df['edad_nan'] = np.where(df['edad'].isnull(), 1, 0)
    #df['mujer_4d_palermo'] = np.where((df['genero'] == 'mujer') & (df['tipo_de_sala'] == '4d') & (df['nombre_sede'] == 'fiumark_palermo'), 1, 0)
    for sede in ['fiumark_palermo', 'fiumark_quilmes', 'fiumark_chacarita']:
        for sala in ['normal', '3d', '4d']:
            df[sede+'_'+sala] = np.where((df['tipo_de_sala'] == sala) & (df['nombre_sede'] == sede), 1, 0)
    #Se procesa el titulo de cada encuestado para que lo utilice el 
    #knnimputer para calcular los missing values en la edad.
    ##df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
    #Se borran columnas que no aportan datos significativos
    borrar_columna(df, 'fila', True)
    borrar_columna(df, 'nombre', True)
    borrar_columna(df, 'id_ticket', True)
    df['edad_knn'] = knn_imputer(df)
    #Se encodean las columnas categoricas
    df = one_hot_encoding(df, 'nombre_sede')
    df = one_hot_encoding(df, 'genero', dummy_na=False)
    df = one_hot_encoding(df, 'tipo_de_sala')
    #Se dropea la edad con missing values y se redondean los valores de edad calculados
    df.drop(['edad'],axis=1, inplace=True)
    df.drop(['precio_ticket'],axis=1, inplace=True)
    df = redondear_edades(df, 'edad_knn')
    #Se dropean las columnas titulo e id_usuario que no son utiles
    #df.drop(columns=['id_usuario', 'nombre_sede', 'tipo_de_sala'], inplace=True)
    df.drop(columns=['id_usuario'], inplace=True)
    return df

def procesamiento_rf_3(df):
    #Se indica que columnas tenian edad nula
    df['edad_nan'] = np.where(df['edad'].isnull(), 1, 0)
    #df['mujer_4d_palermo'] = np.where((df['genero'] == 'mujer') & (df['tipo_de_sala'] == '4d') & (df['nombre_sede'] == 'fiumark_palermo'), 1, 0)
    for sede in ['fiumark_palermo', 'fiumark_quilmes', 'fiumark_chacarita']:
        for sala in ['normal', '3d', '4d']:
            df[sede+'_'+sala] = np.where((df['tipo_de_sala'] == sala) & (df['nombre_sede'] == sede), 1, 0)
    #Se procesa el titulo de cada encuestado para que lo utilice el 
    #knnimputer para calcular los missing values en la edad.
    ##df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
    #Se borran columnas que no aportan datos significativos
    borrar_columna(df, 'fila', True)
    borrar_columna(df, 'nombre', True)
    borrar_columna(df, 'id_ticket', True)
    df['edad_knn'] = knn_imputer(df)
    df.drop(['edad'],axis=1, inplace=True)
    df.drop(['precio_ticket'],axis=1, inplace=True)
    df = redondear_edades(df, 'edad_knn')
    #Se dropean las columnas titulo e id_usuario que no son utiles
    df.drop(columns=['id_usuario', 'nombre_sede', 'tipo_de_sala'], inplace=True)
    #df.drop(columns=['id_usuario'], inplace=True)
    return df





















def procesamiento_arboles_discretizer(df):
    #Se indica que columnas tenian edad nula
    #df['edad_nan'] = np.where(df['edad'].isnull(), 1, 0)
    #Se procesa el titulo de cada encuestado para que lo utilice el 
    #knnimputer para calcular los missing values en la edad.
    #df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
    #df['edad_knn'] = knn_imputer(df)
    #Se borran columnas que no aportan datos significativos
    borrar_columna(df, 'fila', True)
    borrar_columna(df, 'nombre', True)
    borrar_columna(df, 'id_ticket', True)
    #Se encodean las columnas categoricas
    df = one_hot_encoding(df, 'nombre_sede')
    df = one_hot_encoding(df, 'genero')
    df = one_hot_encoding(df, 'tipo_de_sala')
    #df = redondear_edades(df, 'edad_knn')
    df.dropna(subset=['edad'], inplace=True)
    df['edad_bins'] = kbins_discretizer(df, 'edad', 10)
    df.dropna(subset=['precio_ticket'], inplace=True)
    df['precio_bins'] = kbins_discretizer(df, 'precio_ticket', 5)
    df.reset_index(inplace=True)
    #Se dropea la edad con missing values y se redondean los valores de edad calculados
    df.drop(['edad'],axis=1, inplace=True)
    #df.drop(['precio_ticket'],axis=1, inplace=True)
    df.drop(['index'],axis=1, inplace=True)
    print(df.head())
    #Se dropean las columnas titulo e id_usuario que no son utiles
    df.drop(columns=['id_usuario'], inplace=True)
    return df

# -------------- FUNCIONES AUXILIARES -----------------

def borrar_columna(df, columna, ip=False):
    if ip:
        df.drop(axis=1, columns=[columna], inplace=ip)
    else:
        return df.drop(axis=1, columns=[columna])

def one_hot_encoding(df, variable, dummy_na=True, drop_first=True):
    return pd.get_dummies(df, columns=[variable], dummy_na=dummy_na, drop_first=drop_first)    

def label_encoding(df, variable):
    le = LabelEncoder()
    return le.fit_transform(df[variable].astype(str))

def redondear_edades(df, columna_edad):
    df[columna_edad] = df[columna_edad].round().astype(int)
    return df


def hashing_encoding(df, cols, data_percent=0.85, verbose=False):
    for i in cols:
        val_counts = df[i].value_counts(dropna=False)
        s = sum(val_counts.values)
        h = val_counts.values / s
        c_sum = np.cumsum(h)
        c_sum = pd.Series(c_sum)
        n = c_sum[c_sum > data_percent].index[0]
        if verbose:
            print("n hashing para ", i, ":", n)
        if n > 0:
            fh = FeatureHasher(n_features=n, input_type='string')
            hashed_features = fh.fit_transform(
                df[i].astype(str).values.reshape(-1, 1)
            ).todense()
            df = df.join(pd.DataFrame(hashed_features).add_prefix(i + '_'))

    return df.drop(columns=cols)


def knn_imputer(df):

    cat_cols = ['tipo_de_sala', 'genero', 'nombre_sede']

    # Aplicamos hashing para las categoricas
    df = hashing_encoding(df, cat_cols)

    # definimos un n arbitrario
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df.add_suffix('_knn')['edad_knn']

def kbins_discretizer(df, col, n_bins=4):
    enc = KBinsDiscretizer(n_bins, encode='ordinal')

    _df = df[[col]].dropna().reset_index(drop=True)
    X_binned = enc.fit_transform(_df)
    return X_binned.astype(int)


