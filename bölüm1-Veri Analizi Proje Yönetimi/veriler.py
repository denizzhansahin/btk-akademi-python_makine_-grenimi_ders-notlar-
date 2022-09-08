#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv") #veri alindi
#print(veriler)
boy = veriler[['boy']]
#print(boy)
boykilo = veriler[['boy','kilo']]
print(boykilo)