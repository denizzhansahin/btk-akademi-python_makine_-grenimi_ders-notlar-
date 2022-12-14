# -*- coding: utf-8 -*-
"""random_forest

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fp8-2k-zHnvVbsebBXRVRqe277Xh99rn
"""

#1.kutuphaneler eklenir
from cgi import test
import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv("maaslar.csv")
#print(veriler)

#data frame dilimleme(slice)
x = veriler.iloc[:,1:2] #tum satırlar ve 0'dan 2'e kadar 2 haric- hepsini al
y = veriler.iloc[:, 2:]
#dataframe nedeni ile grafik sorunu yasamamak icin degerlerini al
#dizi, numpy array donusumu
X = x.values
Y = y.values

#print(x)
#print(y)



#polynomial regression
#dogrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color="blue")
plt.show()

#polynomial regreassion
#dogrusal olmayan nonlinear model olusturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) #ikinci dereceden olustur
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y) #x'e gore y'yi egit
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

poly_reg2 = PolynomialFeatures(degree=4) #ikinci dereceden olustur
x_poly = poly_reg2.fit_transform(X)
print(x_poly)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y) #x'e gore y'yi egit
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg3.predict(poly_reg2.fit_transform(X)),color="blue")
plt.show()

#Tahmin
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
#tahmine varmadan donusum yap
print(lin_reg3.predict(poly_reg2.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg2.fit_transform([[11]])))

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
X_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
Y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR
SVR_reg = SVR(kernel = 'rbf')
SVR_reg.fit(X_olcekli,Y_olcekli)
plt.scatter(X_olcekli,Y_olcekli,color="red")
plt.plot(X_olcekli,SVR_reg.predict(X_olcekli),color="blue")
plt.show()
print(SVR_reg.predict([[11]]))
print(SVR_reg.predict([[6.6]]))

from sklearn.tree import DecisionTreeRegressor
r_dt  = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X), color="blue")
plt.plot(x,r_dt.predict(Z), color = "green")
plt.plot(x,r_dt.predict(K), color="yellow")
plt.plot(r_dt.predict([[11]]))
plt.plot(r_dt.predict([[6.6]]))
plt.show()

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state=0) #n_estimators ne kadar estimator kullanılacak 10 farklı decision tree
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")

plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")