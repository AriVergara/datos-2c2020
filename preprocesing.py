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

#--------- FUNCIONES DE PREPROCESAMIENTO ------------
        
def procesamiento_arboles(df):
    #Se indica que columnas tenian edad nula
    df['edad_nan'] = np.where(df['edad'].isnull(), 1, 0)
    #Se procesa el titulo de cada encuestado para que lo utilice el 
    #knnimputer para calcular los missing values en la edad.
    df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
    df['edad_knn'] = knn_imputer(df)
    #Se borran columnas que no aportan datos significativos
    borrar_columna(df, 'fila', True)
    borrar_columna(df, 'nombre', True)
    borrar_columna(df, 'id_ticket', True)
    #Se encodean las columnas categoricas
    df = one_hot_encoding(df, 'nombre_sede')
    df = one_hot_encoding(df, 'genero')
    df = one_hot_encoding(df, 'tipo_de_sala')
    #Se dropea la edad con missing values y se redondean los valores de edad calculados
    df.drop(['edad'],axis=1, inplace=True)
    df = redondear_edades(df, 'edad_knn')
    #Se dropean las columnas titulo e id_usuario que no son utiles
    df.drop(columns=['titulo', 'id_usuario'], inplace=True)
    return obtener_sets(df)
    
#-------------- FUNCIONES AUXILIARES -----------------
    
def obtener_sets(df):
    X = df.drop(columns=['volveria'])
    y = df.volveria
    return train_test_split(X, y, test_size=0.2, random_state=0)
    
def borrar_columna(df, columna, ip=False):
    if ip:
        df.drop(axis=1, columns=[columna], inplace=ip)
    else:
        return df.drop(axis=1, columns=[columna])

def one_hot_encoding(df, variable):
    return pd.get_dummies(df, columns=[variable], dummy_na=True, drop_first=True)    

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

    cat_cols = ['tipo_de_sala', 'genero', 'nombre_sede', 'titulo']

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
