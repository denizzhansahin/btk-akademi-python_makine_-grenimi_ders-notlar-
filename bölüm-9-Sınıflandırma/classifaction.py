# -*- coding: utf-8 -*-
"""classifaction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kEZ_svZ2IYtbzBFBkDedXlOj3BShKnej
"""

#!/usr/bin/env python3

#1.kutuphaneler eklenir
from cgi import test
import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veriler alindi
#2.veri onisleme
#2.1 veri yukleme
veriler = pd.read_csv("veriler.csv")
#test

x = veriler.iloc[:,1:4].values #bagimsiz degisken
y = veriler.iloc[:,4:].values #bagimli degisken


#verileri egitim ve test icin bol ve verilerin olceklenmesi
from sklearn.model_selection import train_test_split #belli bir yere kadar deneme belli bir yerden sonra test olacak
#x bağımsız, y bapımlı değişken

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)

#verileri olcekle
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train) #egit ve uygula
X_test = sc.transform(x_test) #ogrenmeden uygulama

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

#tahmin
y_pred = logr.predict(X_test)#tahmin et test verisi ile
print(x_test)
print(y_pred)

print(y_test)#gercek bilgi