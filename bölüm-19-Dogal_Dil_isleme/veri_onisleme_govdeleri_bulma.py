# -*- coding: utf-8 -*-
"""govdeleri_bulma.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iG8_4jN7XRA1RFmmrenC_4t6fe39TjkL
"""

from nltk.stem.snowball import PorterStemmer
import numpy as np
import pandas as pd

yorumlar = pd.read_csv('yorumlar.csv')

import re
#alfa numerik karekter filtrelendi
"""
yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][6]) #yoruum alıp, fitrele yaptı ve değiştirdi, boşuk karekteri değişirse siler
print(yorum)
"""
"""
yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][0]) #yoruum alıp, fitrele yaptı ve değiştirdi
print(yorum)
yorum = yorum.lower()#yazının tamamı kucuk harfe donusur
yorum = yorum.split()#liste olur
print(yorum)
"""
import nltk
nltk.download('stopwords') #ingilizce stopwords indirdi
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer #kelime gövedeleri için
ps = PorterStemmer()

#govdelere ayırdı
"""
yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] #liste tanımladık ve ps kullandık
print(yorum)

#tekrar cumle yap
yorum = ' '.join(yorum)
"""

derlem = []
#yorumlardaki eleman sayısı 1000 tane var
for i in range(1000):
  yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
  yorum = yorum.lower()
  yorum = yorum.split()
  yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
  yorum = ' '.join
  derlem.append(yorum)