# -*- coding: utf-8 -*-
"""regresyon_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Kl3YBQdPhwBaA7rkhTyXPLMOBwBboxC0
"""

#1.kutuphaneler eklenir
from cgi import test
import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veriler alindi
#2.veri onisleme
#2.1 veri yukleme
veriler = pd.read_csv("satislar.csv")
#test
#print(veriler)

#veri onisleme
aylar = veriler[['Aylar']]
#print(aylar)
satislar = veriler[['Satislar']]
#print(satislar)
satislar2 =veriler.iloc[:,:1].values
#print(satislar2)

#verileri egitim ve test icin bol ve verilerin olceklenmesi
from sklearn.model_selection import train_test_split #belli bir yere kadar deneme belli bir yerden sonra test olacak
#x bağımsız, y bapımlı değişken

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

#farkli dünyadaki veriler aynı dünyaya ekle
#verileri olcekle
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
#print(X_train)
#print(X_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
#regresyon- linear regression-model inşası
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train) #modeli oluştur