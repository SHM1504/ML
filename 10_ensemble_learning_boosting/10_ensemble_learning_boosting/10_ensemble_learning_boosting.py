#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:00:33 2023

@author: d

Sources: https://www.analyticsvidhya.com/blog/2021/04/best-boosting-algorithm-in-machine-learning-in-2021/
         https://medium.com/analytics-vidhya/catboost-101-fb2fdc3398f3
Dataset: https://www.kaggle.com/datasets/uciml/mushroom-classification?select=mushrooms.csv
"""

"""Boosting"""

# Boosting is a method used in machine learning to improve the accuracy
# of a model by combining the predictions of multiple weaker models.
# The idea is to train a sequence of models in a stage-wise manner,
# where each model tries to correct the mistakes of the previous model.
# The final prediction is made by combining the predictions of all
# the models in the sequence, typically through a weighted majority vote.
# One of the most popular boosting algorithms is called AdaBoost.

#%% Importing libraries

import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import xgboost as xgb

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#%% Boosting Intuition

im = img.imread('boosting_intuition.png')
plt.figure(dpi=1200)
plt.axis('off')
plt.imshow(im)
plt.show()

#%%

""" Getting Data """

df = pd.read_csv('./datasets/mushrooms.csv')
df = df.sample(frac = 1)

print("Dataframe:")
print("--------")
print(df)
print("-------------------------------------------------------------------")
print("Columns:")
print("--------")
print(df.columns)
print("-------------------------------------------------------------------")

for label in df.columns: 
    df[label]=LabelEncoder().fit(df[label]).transform(df[label])

print("Info:")
print("-----")
print(df.info())

X = df.drop(['class'], axis=1)
    
y = df['class']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%

""" Gradient Boosting """

# Gradient boosting is a method used to improve the accuracy of a model
# by training an ensemble of decision trees. It uses gradient descent
# to minimize the loss function by fitting a new tree at each iteration,
# and adding it to the ensemble with a weight determined by the current loss.
# The final prediction is made by combining the predictions of all the
# decision trees in the ensemble.

cl = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
cl.fit(X_train, y_train)

y_pred = cl.predict(X_test)

predictions = accuracy_score(y_test, y_pred)

print("Accuracy of Gradient Boosting: ", predictions)

#%%%%

""" AdaBoost """

# AdaBoost, short for Adaptive Boosting, is a boosting algorithm used to
# improve the accuracy of a model by combining the predictions of multiple
# weaker models. It works by fitting a sequence of models to the data,
# where each model tries to correct the mistakes of the previous model.
# The final prediction is made by combining the predictions of all the
# models in the sequence, typically through a weighted majority vote.

dtree = DecisionTreeClassifier()

cl = AdaBoostClassifier(n_estimators=100, estimator=dtree, learning_rate=1)
cl.fit(X_train,y_train)

y_pred = cl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of AdaBoost: ", accuracy)

#%%%

""" XGBoost - improvised version of the gradient boosting algorithm """

# XGBoost (eXtreme Gradient Boosting) is an open-source implementation
# of the gradient boosting algorithm. It is designed to be efficient and
# scalable, making it a popular choice for large-scale machine learning
# tasks such as classification and regression. XGBoost is known for its
# high performance, fast training times, and ability to handle missing
# values and large datasets.

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=1, max_depth=1) # learning_rate=0.1, max_depth=3
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of XGBoost: ", accuracy)

#%%

""" Catboost - Works well on Heterogeneous data. """

# CatBoost is a gradient boosting algorithm that is specifically designed to
# handle categorical features in the dataset without any preprocessing
# required. It uses a novel technique called "permutation-based algorithm"
# to handle categorical variables by calculating the optimal split points on
# the categorical feature rather than the one-hot encoding method. It also has
# built-in handling of missing values, and it is efficient for large datasets
# and has built-in visualization tools to help understand the model.

cat_model = CatBoostClassifier(
    iterations = 1000, # 1000 are ideal
    loss_function ='MultiClass',
    bootstrap_type = "Bayesian",
    eval_metric = 'MultiClass',
    leaf_estimation_iterations = 100,
    random_strength = 0.5,
    depth = 7,
    l2_leaf_reg = 5,
    learning_rate=0.1,
    bagging_temperature = 0.5,
    task_type = "CPU",
)

# training the model
cat_model.fit(X_train, y_train)

# predicting the model output
y_pred_cat = cat_model.predict(X_test)
# printing the accuracy of the tuned model
print("Accuracy of CatBoost: ", accuracy_score(y_test, y_pred_cat))

# confusion metrics of the LightGBM and plotting the same
confusion_matrix_LightGBM = confusion_matrix(y_test, y_pred_cat)
print(confusion_matrix_LightGBM)
