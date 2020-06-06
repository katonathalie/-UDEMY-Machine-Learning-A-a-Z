#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:41:53 2020

@author: nathaliekato
"""


import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]

# apagar a coluna
base.drop('age', 1, inplace=True)

# pagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)

# substituir com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)