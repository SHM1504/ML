# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:47:21 2025

@author: me
"""


import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

#%%

glass = pd.read_csv("glassClass.csv")
print(glass.head())
print(glass.shape)
print(glass.columns)

#%%

le = LabelEncoder()
le.fit(glass["quality"])
glass['quality'] = le.transform(glass['quality'])
print(list(le.classes_))

#%%

X = glass[['Unnamed: 0', 'fixed.acidity', 'volatile.acidity', 'citric.acid',
       'residual.sugar', 'chlorides', 'free.sulfur.dioxide',
       'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

g_matrix = np.array(X)

cluster_model = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward' )
cluster_labels = cluster_model.fit_predict(g_matrix)
print(cluster_labels)

glass['pred'] = cluster_labels
print(glass.head())

#%%

from sklearn import metrics
acc = sm.accuracy_score(glass.quality,cluster_model.labels_)

metr= metrics.adjusted_rand_score(glass.quality,cluster_model.labels_)

print(acc)
print(metr)
#%%

cg = sns.clustermap(glass)
plt.show()

#%%

cg= sns.clustermap(glass.corr())

