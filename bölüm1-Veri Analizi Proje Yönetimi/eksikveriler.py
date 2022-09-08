#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veriler alindi
veriler = pd.read_csv("eksikveriler.csv")
print(veriler)
boy = veriler[['boy']]
print(boy)
boykilo = veriler[['boy','kilo']]
print(boykilo)

#eksik verileri doldur
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4]) #degerleri ogren
Yas[:,1:4] = imputer.transform(Yas[:,1:4]) #degerleri degistir
print(Yas)