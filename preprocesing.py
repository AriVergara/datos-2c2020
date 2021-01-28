import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder
)

def borrar_columna_fila(df):
	return df.drop(axis=1, columns=["fila"], inplace=False)

def borrar_columna_nombre(df):
	return df.drop(axis=1, columns=["nombre"], inplace=False)

def borrar_columna_id_ticket(df):
	return df.drop(axis=1, columns=["id_ticket"], inplace=False)

def one_hot_encoding(df, variable):
	return pd.get_dummies(df, columns=[variable], dummy_na=True, drop_first=True)

def label_encoding(df, variable):
	le = LabelEncoder()
	return le.fit_transform(df[variable].astype(str))

def redondear_edades(df, columna_edad):
	df[columna_edad] = df[columna_edad].round().astype(int)
	return df

def procesar_titulos(df):
    df['titulo'] = df.nombre.str.split(expand=True).iloc[:,0]
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
