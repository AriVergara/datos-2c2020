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
        self.moda_nombre_sede = ""
    
    def fit(self, X, y=None):
        self.moda_nombre_sede = X["nombre_sede"].astype(str).mode()[0]
        self.mean_edad = X["edad"].mean()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].fillna(self.moda_nombre_sede).astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[:, "fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        X["edad_bins"] = X["edad"].apply(self._bins_segun_edad)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        
        X["nombre_sede_isna"] = X["nombre_sede"].isna().astype(int)
        X['nombre_sede'] = X['nombre_sede'].fillna(self.moda_nombre_sede)
        X['nombre_sede'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero.transform(X['genero'].astype(str))

        X["precio_ticket_bins"] = X["precio_ticket"].apply(self._bins_segun_precio)
        #X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        return X
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3
    
    def _bins_segun_edad(self, edad): 
        if edad <= 18:
            return 1
        if 18 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        if 40 < edad <= 70:
            return 4
        return 5


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
        X = X.copy()
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
        X = X.copy()
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
        self.scaler.fit(self._transform(X))
        return self
    
    def _transform(self, X):
        X = X.copy()
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
        
        return X

    def transform(self, X):
        X = self._transform(X)
        return self.scaler.transform(X)
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3

#XGBoost

class PreprocessingXGBoost(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (id_usuario, id_ticket, nombre).
    -Encodea variables categóricas mediante OneHotEncoding (genero, nombre_sala, tipo_de_sala)
    -Agrega las columnas edad_isna y fila_isna
    -Completa los missing values de la columna edad con la mediana
    -Se crean bins de tamaño 2 para la edad y el precio_ticket
    """
    def __init__(self):
        super().__init__()
        self._bins_edad = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile")
        self._bins_precio_ticket = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile")
    
    def fit(self, X, y=None):
        self._mediana_edad = X.edad.median()
        self._mediana_precio_ticket = X.precio_ticket.median()
        self._bins_edad.fit(X[["edad"]].fillna(X.edad.median()))
        self._bins_precio_ticket.fit(X[["precio_ticket"]].fillna(X.precio_ticket.median()))
        return self

    def transform(self, X):
        X = X.copy()
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self._mediana_edad)
        X["edad_bins"] = pd.DataFrame(self._bins_edad.transform(X[["edad"]]))[0]
        #X = X.drop(columns=["edad"], axis=1, inplace=False)
        #X["edad_limite_inferior"] = X["edad"].apply(self.setear_limite_inferior_bins, args=(self._bins_edad,))
        #X["edad_limite_superior"] = X["edad"].apply(self.setear_limite_superior_bins, args=(self._bins_edad,))
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)

        X["precio_ticket"] = X["precio_ticket"].fillna(self._mediana_precio_ticket)
        X["precio_ticket_bins"] = pd.DataFrame(self._bins_precio_ticket.transform(X[["precio_ticket"]]))[0]
        #X["precio_ticket_limite_inferior"] = X["precio_ticket"].apply(self.setear_limite_inferior_bins, args=(self._bins_precio_ticket,))
        #X["precio_ticket_limite_superior"] = X["precio_ticket"].apply(self.setear_limite_superior_bins, args=(self._bins_precio_ticket,))
        #X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        return X
    
    def setear_limite_inferior_bins(self, selected_bin, discretizer):
        limites = discretizer.bin_edges_[0]
        cantidad_bins = discretizer.n_bins_[0]
        for i in range(cantidad_bins):
            if limites[i] <= selected_bin < limites[i+1]:
                return limites[i]
            
    def setear_limite_superior_bins(self, selected_bin, discretizer):
        limites = discretizer.bin_edges_[0]
        cantidad_bins = discretizer.n_bins_[0]
        for i in range(cantidad_bins):
            if limites[i] <= selected_bin < limites[i+1]:
                return limites[i+1]
        return limites[cantidad_bins]


class PreprocessingXGBoost2(BaseEstimator, TransformerMixin):
    """
    -Elimina columnas sin infromación valiosa (id_usuario, id_ticket, nombre).
    -Encodea variables categóricas mediante OneHotEncoding (genero, nombre_sala, tipo_de_sala)
    -Agrega las columnas edad_isna y fila_isna
    -No completa missing values
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["fila_isna"] = X["fila"].isna().astype(int)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)

        X["edad_isna"] = X["edad"].isna().astype(int)
        
        X = pd.get_dummies(X, columns=['genero'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['tipo_de_sala'], dummy_na=True, drop_first=True) 
        
        X = pd.get_dummies(X, columns=['nombre_sede'], dummy_na=True, drop_first=True)
        
        return X
    
# Naive Bayes

class PreprocessingCategoricalNB1(BaseEstimator, TransformerMixin):
    """
        -Elimina columnas sin información valiosa (fila, id_usuario, id_ticket) y con valores 
            continuos o discretos(parientes, amigos, edad y precio_ticket).
        -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
        -Agrega columnas edad_isna, fila_isna
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self._moda_nombre_sede = ""
    
    def fit(self, X, y=None):
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        self._moda_nombre_sede = self._obtener_moda_nombre_sede(X)
        return self

    def transform(self, X):
        X = X.copy()
        X["edad_isna"] = X["edad"].isna().astype(int)
        X["fila_isna"] = X["fila"].isna().astype(int)
        
        X['tipo_de_sala_encoded'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        X = X.drop(columns=["tipo_de_sala"], axis=1, inplace=False)
        
        X['genero_encoded'] = self.le_genero.transform(X['genero'].astype(str))
        X = X.drop(columns=["genero"], axis=1, inplace=False)
        
        X["nombre_sede_isna"] = X["nombre_sede"].isna().astype(int)
        X["nombre_sede"] = X["nombre_sede"].fillna(self._moda_nombre_sede)
        X['nombre_sede_encoded'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        X = X.drop(columns=["nombre_sede"], axis=1, inplace=False)
        
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["amigos"], axis=1, inplace=False)
        X = X.drop(columns=["parientes"], axis=1, inplace=False)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        
        return X
    
    def _obtener_moda_nombre_sede(self, X):
        return X.nombre_sede.value_counts().index[0]
    
class PreprocessingCategoricalNB2(BaseEstimator, TransformerMixin):
    """
        -Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket) y con valores 
            continuos o discretos(parientes, amigos, edad y precio_ticket).
        -Encodea variables categóricas mediante LabelEncoding (genero, nombre_sala, tipo_de_sala)
        -Transforma en bins la edad y el precio_ticket.
    """
    def __init__(self):
        super().__init__()
        self._valores_fila = []
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
    
    def fit(self, X, y=None):
        self._valores_fila = X.fila.dropna().unique()
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        self._moda_nombre_sede = self._obtener_moda_nombre_sede(X)
        return self

    def transform(self, X):
        X = X.copy()
        X['tipo_de_sala_encoded'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        X = X.drop(columns=["tipo_de_sala"], axis=1, inplace=False)
        
        X['genero_encoded'] = self.le_genero.transform(X['genero'].astype(str))
        X = X.drop(columns=["genero"], axis=1, inplace=False)
        
        X["nombre_sede"] = X["nombre_sede"].fillna(self._moda_nombre_sede)
        X['nombre_sede_encoded'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        X = X.drop(columns=["nombre_sede"], axis=1, inplace=False)
        
        X['edad_bins'] = X['edad'].apply(self._bins_segun_edad_cuantiles)
        X = X.drop(columns=["edad"], axis=1, inplace=False)
        
        X['precio_ticket_bins'] = X['precio_ticket'].apply(self._bins_segun_precio)
        X = X.drop(columns=["precio_ticket"], axis=1, inplace=False)
        
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["amigos"], axis=1, inplace=False)
        X = X.drop(columns=["parientes"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        
        return X
    
    def reemplazar_valores_de_fila_desconocidos(self, fila):
        if fila not in self._valores_fila:
            return np.nan()
        return fila
    
    def _bins_segun_precio(self, valor):
        if valor == 1:
            return 1
        if 2 <= valor <= 3:
            return 2
        return 3
    
    def _bins_segun_edad_cuantiles(self, edad):
        if np.isnan(edad):
            return 0
        if edad <= 23:
            return 1
        if 23 < edad <= 31:
            return 2
        if 31 < edad <= 41:
            return 3
        return 4
    
    def _obtener_moda_nombre_sede(self, X):
        return X.nombre_sede.value_counts().index[0]
    

class PreprocessingGaussianNB1(BaseEstimator, TransformerMixin):
    """
        - Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket) y con valores 
            categoricos (genero, fila, tipo_de_sala, nombre_sede).
        - Se agrega la columna acompaniantes y se eliminan parientes y amigos.
    """
    def __init__(self):
        super().__init__()
        self._mean_edad = 0
    
    def fit(self, X, y=None):
        self._mean_edad = X["edad"].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["edad"] = X["edad"].fillna(self._mean_edad)
        X["acompaniantes"] = X["parientes"] + X["amigos"]
        
        X = X.drop(columns=["parientes"], axis=1, inplace=False)
        X = X.drop(columns=["amigos"], axis=1, inplace=False)
        X = X.drop(columns=["genero"], axis=1, inplace=False)
        X = X.drop(columns=["tipo_de_sala"], axis=1, inplace=False)
        X = X.drop(columns=["nombre_sede"], axis=1, inplace=False)
        X = X.drop(columns=["fila"], axis=1, inplace=False)
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        
        return X

# -------------- TRANSFORMERS AUXILIARES -----------------
# Refactor para poder hacer que cada uno de los Preprocessor sea una función que devuelve un Pipeline 
#con los preprocesamientos adecuados


class EliminarFilasSinInformacionTransformer(BaseEstimator, TransformerMixin):
    """
        Elimina columnas sin infromación valiosa (fila, id_usuario, id_ticket).
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=["id_usuario"], axis=1, inplace=False)
        X = X.drop(columns=["nombre"], axis=1, inplace=False)
        X = X.drop(columns=["id_ticket"], axis=1, inplace=False)
        return X
    
class LabelEncoderCategoricas(BaseEstimator, TransformerMixin):
    """
        - Aplica label encoder sobre las variables categóricas (nombre_sede, genero, tipo_de_sala)
    """
    def __init__(self):
        super().__init__()
        self.le_tipo_sala = LabelEncoder()
        self.le_nombre_sede = LabelEncoder()
        self.le_genero = LabelEncoder()
        self.moda_nombre_sede = ""
    
    def fit(self, X, y=None):
        self.moda_nombre_sede = X["nombre_sede"].astype(str).mode()[0]
        self.le_tipo_sala.fit(X['tipo_de_sala'].astype(str))
        self.le_nombre_sede.fit(X['nombre_sede'].fillna(self.moda_nombre_sede).astype(str))
        self.le_genero.fit(X['genero'].astype(str))
        return self

    def transform(self, X):
        X = X.copy()nplace=False)
        
        X["nombre_sede_isna"] = X["nombre_sede"].isna().astype(int)
        X['nombre_sede'] = X['nombre_sede'].fillna(self.moda_nombre_sede)
        X['nombre_sede'] = self.le_nombre_sede.transform(X['nombre_sede'].astype(str))
        
        X['tipo_de_sala'] = self.le_tipo_sala.transform(X['tipo_de_sala'].astype(str))
        
        X['genero'] = self.le_genero.transform(X['genero'].astype(str))
        
        return X
    
class EdadTransformer(BaseEstimator, TransformerMixin):
    """
        - Completa los missing values de la columna edad con la media
        - Convierte en bins los valores de la columna edad.
        
    """
    def __init__(self, drop_column=False):
        super().__init__()
        self.mean_edad = 0
        self.drop_column = drop_column
    
    def fit(self, X, y=None):
        self.mean_edad = X["edad"].mean()
        return self

    def transform(self, X):
        X = X.copy()

        X["edad_isna"] = X["edad"].isna().astype(int)
        X["edad"] = X["edad"].fillna(self.mean_edad)
        X["edad_bins"] = X["edad"].apply(self._bins_segun_edad)
        if self.drop_column:
            X = X.drop(columns=["edad"], axis=1, inplace=False)
        return X
    
    def _bins_segun_edad(self, edad): 
        if edad <= 18:
            return 1
        if 18 < edad <= 30:
            return 2
        if 30 < edad <= 40:
            return 3
        if 40 < edad <= 70:
            return 4
        return 5