# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:10:22 2025

@author: swett
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

#%%

df= pd.read_csv("Ralph_Draft/banknotes.csv", sep =',')

print(df.head())
print(df.columns)
#%%



def conterfeit(bn_id):
    bn_id = bn_id.replace('BN','')
    id = int(bn_id)
    if id in range(0,101):
        counterfeit = 0
    elif id in range(101,203):
        counterfeit = 1
    else:
        counterfeit = None
        
    return counterfeit
        

df['counterfeit'] = df['ID'].apply(lambda x: conterfeit(x))

print(df.head())
#%%

df['ID'] = df['counterfeit']
df = df.drop('counterfeit', axis = 1)

print(df.head())


#%%
df.rename(columns ={'ID': 'counterfeit'}, inplace = True)
print(df.head())

df.to_csv('Ralph_Draft/banknotes_neu.csv')
















