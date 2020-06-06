#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:41:53 2020

@author: nathaliekato
"""


import pandas as pd

base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]

# apagar a coluna
# base.drop('age', 1, inplace=True)

# pagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)