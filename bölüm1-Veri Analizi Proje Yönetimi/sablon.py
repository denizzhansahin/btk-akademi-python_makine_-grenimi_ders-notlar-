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
veriler = pd.read_csv("eksikveriler.csv")
#test
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

#encoder , kategorik ordinal deger -> Numeric deger
ulke = veriler.iloc[:,0:1].values #bir kolonu alır veilen indexlerle bir deger alir
print(ulke)

#donusum
from sklearn import preprocessing
le = preprocessing.LabelEncoder()#her bir degere 0 ve 1 gibi deger verir
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])# verileri sayıya donusur
print(ulke)

ohe = preprocessing.OneHotEncoder()#kolon basliklari ve etiketleri tasimak ve etiket altina 0 ve 1 gibi deger verir, oraya ait mi oldugunu belirt ya da belirtme
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#numpy dizilierine kolon basligi ve dizi ekle, aslinda bir donusum var, dataframe küekle
sonuc = pd.DataFrame(data=ulke, index = range(22), columns=['fr','tr','us'])
print(sonuc)
sonuc2= pd.DataFrame(data=Yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc2)
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)
sonuc3=pd.DataFrame(data=cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc3)
#dataframe birlestir
s=pd.concat([sonuc, sonuc2], axis=1) #birlestir concat, sifirinci satırı alır baslat axis
print(s)
s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verileri egitim ve test icin bol ve verilerin olceklenmesi
from sklearn.model_selection import train_test_split #belli bir yere kadar deneme belli bir yerden sonra test olacak
#x bağımsız, y bapımlı değişken

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

#farkli dünyadaki veriler aynı dünyaya ekle
#verileri olcekle
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x_train)
print(x_test)