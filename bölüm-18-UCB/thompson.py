# -*- coding: utf-8 -*-
"""thompson.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pSxrqrhrv41uG-QWXxUyRdeDNHTE8cu5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#random selection
"""
import random
N = 10000
d = 10
toplam = 0
secilenler = []
for i in range(0,N):
  ad = random.randrange(d) #rastgele bir sayı üret
  secilenler.append(ad)
  odul = veriler.values[ad] #n.satir 1 ise odul var yoksa 0 alır , bu calismadi odul = veriler.values[n,ad]
  toplam = toplam + odul

print(toplam)
plt.hist(secilenler)
plt.show()
"""

import random
#ucb
N = 10000 #tıklama
d = 10 #ilan
#ri(n)
#ni(n)
toplam = 0 #toplam ödül
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for i in range(1,N):
  ad = 0 #secilen ilan
  max_th = 0
  for i in range(0,10):
    rasbeta = random.betavariate(birler[i] +1, sifirlar[i]+1)
    if rasbeta > max_th:
      max_th = rasbeta
      ad = i
  secilenler.append(ad)
  odul = veriler.values[3,ad] #3 yerine n yaz ama n nereden geldi bulamadım
  if odul == 1:
    birler[ad] = birler[ad]+1
  else:
    sifirlar[ad] = sifirlar[ad]+1
  toplam = toplam + odul

print("Toplam Ödül: ")
print(toplam)

plt.hist(secilenler)
plt.show()