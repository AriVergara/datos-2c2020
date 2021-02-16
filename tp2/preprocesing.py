import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)
from sklearn.base import BaseEstimator, TransformerMixin


# --------- FUNCIONES DE PREPROCESAMIENTO ------------

RANDOM_STATE = 117

TEST_SIZE = 0.2


class PreprocessingLE(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad con la media
    -Convierte en bins los valores de las columnas edad y precio_ticket.
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self.mean_edad = 0
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        X["edad_bins"] = X["edad"].apply(self._bins_segun_edad_2)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        X['nombre_sede'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero.transform(X['genero'].astype(str))

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3
    
    def _bins_segun_edad(self, edad): 
        if edad <= 20:
            return 1
        if 20 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        return 4
    
    def _bins_segun_edad_2(self, edad): 
        if edad <= 18:
            return 1
        if 18 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        if 40 < edad <= 70:
            return 4
        return 5
    
class PreprocessingLE_2(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad con la media.
    -Convierte en bins los valores de la columna precio_ticket.
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self.mean_edad = 0
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)

        X['nombre_sede'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero.transform(X['genero'].astype(str))

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3


class PreprocessingOHE(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante OneHotEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad con la media.
    -Convierte en bins los valores de la columna precio_ticket.
    """
    def __init__(self):
        super().__init__()
        self.mean_edad = 0
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()
        return self

    def transform(self, X):
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3

class PreprocessingOHE_2(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante OneHotEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad mediante KNNImputer.
    -Convierte en bins los valores de la columna precio_ticket.
    """
    def __init__(self):
        super().__init__()
        self.imputer_edad = KNNImputer(n_neighbors=2, weights="uniform")
    
    def fit(self, X, y=None):
        cat_cols = ['tipo_de_sala', 'genero', 'nombre_sede']
        X = hashing_encoding(X, cat_cols)
        self.imputer_edad.fit(X)
        return self

    def transform(self, X):
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)["edad"]
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3
    
class PreprocessingSE(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante OneHotEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad con la media.
    -Escala los valores numéricos (edad, precio_ticket, parientes y amigos) a media 0 y desvio estandar 1 con StandardScaler.
    """
    def __init__(self):
        super().__init__()
        self.mean_edad = 0
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()
        self._fit_scaler(X)
        return self
    
    def _fit_scaler(self, X):
        X = X.copy()
        X["edad"] = X["edad"].fillna(self.mean_edad)
        self.scaler.fit(X[["edad", "precio_ticket", "parientes", "amigos"]])
        
    def transform(self, X):
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        
        X[["edad", "precio_ticket", "parientes", "amigos"]] = self.scaler.transform(X[["edad", "precio_ticket", "parientes", "amigos"]])
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)

        return X
    
class PreprocessingSE_2(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
    -Completa los missing values de la columna edad con la media.
    -Escala los valores a media 0 y desvio estandar 1 con StandardScaler.
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self.mean_edad = 0
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()   
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        self._fit_scaler(X)
        return self
    
    def _fit_scaler(self, X):
        X = X.copy()
        X["edad"] = X["edad"].fillna(self.mean_edad)
        self.scaler.fit(X[["edad", "precio_ticket", "parientes", "amigos"]])

    def transform(self, X):
        X = X.copy()
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X[["edad", "precio_ticket", "parientes", "amigos"]] = self.scaler.transform(X[["edad", "precio_ticket", "parientes", "amigos"]])
        
        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        
        X['nombre_sede'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero.transform(X['genero'].astype(str))
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3

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


