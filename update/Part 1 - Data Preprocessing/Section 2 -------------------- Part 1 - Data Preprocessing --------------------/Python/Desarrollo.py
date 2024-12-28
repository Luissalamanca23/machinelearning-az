#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:57:13 2024

@author: luissalamanca
"""

## Importacion de librerias
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values


# Tratamiento de los NAs
from sklearn.impute import SimpleImputer
# mean - promedio
imputer = SimpleImputer(missing_values=np.nan, strategy = "mean") # Promedio de las columnas remplasan los nan
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Importar herramientas para codificación de variables
from sklearn.compose import ColumnTransformer  # Importar ColumnTransformer para aplicar transformaciones a columnas específicas

# Crear un ColumnTransformer que aplica OneHotEncoder a la columna 0 (categorías)
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],  # 'encoder' es el nombre de la transformación, aplica OneHot a la columna 0
    remainder='passthrough'  # Las demás columnas permanecen sin cambios
)

# Transformar la matriz X aplicando OneHotEncoder a la columna 0
X = column_transformer.fit_transform(X)

# Codificar la variable dependiente (y) con LabelEncoder
laberencoder_y = LabelEncoder()  # Crear una instancia de LabelEncoder
y = laberencoder_y.fit_transform(y)  # Transformar y en valores numéricos (por ejemplo, 0 y 1 para clasificación binaria)
