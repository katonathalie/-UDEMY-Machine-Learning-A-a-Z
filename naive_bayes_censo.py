#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:15:00 2020

@author: nathaliekato
"""


import pandas as pd
base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Race", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder = 'passthrough')
previsores = ct.fit_transform(previsores).toarray()

#conversao de valores categóricos
label_encoder_classe = LabelEncoder()
classe = label_encoder_classe.fit_transform(classe)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_train, classe_train)
previsoes = classificador.predict(previsores_test)

from sklearn.metrics import accuracy_score, confusion_matrix
predicao = accuracy_score(previsoes, classe_test)
matriz = confusion_matrix(previsoes, classe_test)
