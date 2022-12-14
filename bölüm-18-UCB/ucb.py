# -*- coding: utf-8 -*-
"""ucb.ipynb

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
import math
#ucb
N = 10000 #tıklama
d = 10 #ilan
#ri(n)
oduller = [0] * d #ilk başta bütün ilanlar sıfır
#ni(n)
toplam = 0 #toplam ödül
tiklamalar = [0] * d #o ana kadarki tıklamalar
secilenler = []
for i in range(0,N):
  ad = 0 #secilen ilan
  max_ucb = 0
  for i in range(0,10):
    if(tiklamalar[i] > 0):
      ortalama = oduller[i] / tiklamalar[i]
      delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
      ucb = ortalama + delta
    else:
      ucb = N*10
    if max_ucb < ucb: #maxtan buyuk bir ucb geldi
      max_ucb = ucb
      ad = i
  secilenler.append(ad)
  tiklamalar[i] = tiklamalar[ad]+1
  odul = veriler.values[ad]
  oduller[ad] = oduller[ad] + odul
  toplam = toplam + odul

print("Toplam Ödül: ")
print(toplam)

plt.hist(secilenler)
plt.show()