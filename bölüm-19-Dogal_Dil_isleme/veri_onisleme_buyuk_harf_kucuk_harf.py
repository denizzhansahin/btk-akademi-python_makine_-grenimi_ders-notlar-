# -*- coding: utf-8 -*-
"""buyuk_harf_kucuk_harf.ipynb

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
yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][0]) #yoruum alıp, fitrele yaptı ve değiştirdi, boşuk karekteri değişirse siler
print(yorum)
"""
yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][6]) #yoruum alıp, fitrele yaptı ve değiştirdi
print(yorum)
yorum = yorum.lower()#yazının tamamı kucuk harfe donusur
yorum = yorum.split()#liste olur
print(yorum)