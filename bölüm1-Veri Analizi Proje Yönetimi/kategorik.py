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

ulke = veriler.iloc[:,0:1].values #bir kolonu alır
print(ulke)

#donusum
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])# verileri sayıya donusur
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index = range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc, sonuc2], axis=1) #birlestir concat, sifirinci satırı alır baslat axis
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)