#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:35:21 2020

@author: nathaliekato
"""


import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


from sklearn.tree import DecisionTreeClassifier, export

classficador = DecisionTreeClassifier(criterion='entropy')
classficador.fit(previsores, classe)
print(classficador.feature_importances_)

export.export_graphviz(classficador,
                       out_file='arvore.dot',
                       feature_names=['hitoria de credito', 'divida', 'garantias', 'renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
#resultado = classficador.predict([[0,0,1,2], [3,0,0,0]])
#print(classficador.classes_)
#print(classficador.class_count_)
#print(classficador.class_prior_)