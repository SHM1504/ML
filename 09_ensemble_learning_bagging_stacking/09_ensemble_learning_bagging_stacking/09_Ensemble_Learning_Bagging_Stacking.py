#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:46:11 2023

@author: d

Quellen:
https://www.kdnuggets.com/2021/05/ensemble-methods-explained-plain-english-bagging.html
https://machinelearningmastery.com/bagging-ensemble-with-python/
https://www.analyticsvidhya.com/blog/2021/08/ensemble-stacking-for-machine-learning-and-deep-learning/
"""

"""
=============================================================================
Ensemble Learning (Bagging + Stacking) 

"The Whole is Greater than the Sum of its Parts" — Aristotle
=============================================================================

Ensemble Learning ist ein maschinelles Lernverfahren, bei dem mehrere Modelle 
kombiniert werden, um bessere Vorhersagen oder Entscheidungen zu treffen. 
Statt ein einzelnes Modell zu verwenden, werden mehrere Modelle erstellt und 
ihre Vorhersagen aggregiert, um eine robustere und leistungsfähigere Vorhersage 
zu erzielen.

Der Grundgedanke hinter Ensemble Learning basiert auf der Schwarmintelligenz. 
Indem verschiedene Modelle mit unterschiedlichen Stärken und Schwächen 
kombiniert werden, kann das Ensemble als Ganzes eine höhere 
Vorhersagegenauigkeit erzielen als es ein einzelnes Modell alleine könnte.

Es gibt verschiedene Ansätze für Ensemble Learning, darunter:

1. Bagging (Bootstrap Aggregating): Hierbei werden mehrere Modelle auf 
  zufälligen Teilmengen des Trainingsdatensatzes trainiert und ihre Vorhersagen 
  aggregiert. Das bekannteste Beispiel ist der Random Forest, bei dem 
  Entscheidungsbäume kombiniert werden.

2. Stacking: Hierbei werden mehrere Modelle als Schichten (Layers) angeordnet. 
  Die Vorhersagen der vorhergehenden Schichten dienen als Eingabe für die
  nächsten Schichten. Das Modell in der letzten Schicht aggregiert die
  Vorhersagen der vorletzten Schicht mit Hilfe von Kreuzvalidierung,
  um die endgültige Vorhersage zu erstellen.

3. Boosting: Dieser Ansatz baut iterative Modelle auf, bei denen jedes Modell 
  versucht, die Fehler des vorherigen Modells zu korrigieren. Das bekannteste 
  Beispiel ist der Gradient Boosting Algorithmus, bei dem schwache Modelle
  (z. B. Entscheidungsbäume) nacheinander trainiert werden, um den Fehler zu 
  minimieren.
  
  - Trainingsdaten bekommen zufällige Gewichte -> erstes Modell wird trainert
    -> fehlerhafte Datenpunkte bekommen höheres Gewicht, richtige ein 
    niedrigeres -> zweites Modell wird trainiert -> ...

Ensemble Learning kann die Vorhersagegenauigkeit verbessern, die Stabilität 
erhöhen, Overfitting reduzieren und die Robustheit gegenüber Ausreißern 
verbessern. Es ist besonders effektiv, wenn die einzelnen Modelle 
unterschiedliche Fehlerquellen haben oder auf unterschiedliche Aspekte 
der Daten fokussiert sind.

Drei Voraussetzungen für das Ensemble Learning:
    1. Modelle voneinander unabhängig
    2. Jedes Modell muss leicht besser vorhersagen als zufälliges Raten
    3. Alle Modelle haben ähnliche Performance
    
Dann sollte das Hinzufügen von mehr Modellen die Performance des Systems 
verbessern.
-> Verringert die Varianz und bekämpft das Overfitting
-> allgemeine Muster werden besser gelernt
"""

#%% Imports

import matplotlib.image as img
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost

#%% Bild 1

im = img.imread('09_Ensemble_01.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()

#%%

"""
=============================================================================
Bagging
=============================================================================

- große Anzahl von schwachen Lernern wird kombiniert um die gleiche Aufgabe zu
  lernen.
- "bagging" -> Bootstrap + AGGregatING
- Jeder "schwache Lerner" (-> "base estimators") wird auf einer Teilmenge der 
  Trainingsdaten trainiert
- diese Teilmengen werden mit Bootstrapping erstellt: Teilmenge wird erstellt, 
  in der auch Duplikate von den darin enthaltenen Items vorkommen dürfen
- Ziel: Möglichst viele unterschiedliche "Trainingssätze" zu generieren.
- Bootstrapping garantiert Unabhängigkeit und Unterschiedlichkeit
- Base Estimators performen nur leicht besser als der Zufall
- Beispiel: "shallow decision tree", bei dem jeder Baum auf eine maximale Tiefe
  begrenzt ist.
- Diese Vorhersagen werden dann zu einer kombiniert (Mittelwertbildung)
- Bagging kann sowohl bei Klassifizierung als auch bei Regression verwendet 
  werden: Für Regression nimmt man am Ende ein "soft voting" (Durchschnitt), 
  bei Klassifizierung eine Mehrheitsentscheidung ("hard voting")
"""

#%% Sklearn - Bagging (Beispiele aus zweiter Quelle)

# evaluate bagging ensemble for regression

# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15,
                       noise=0.1, random_state=5)

# define the model
model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)

# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                           cv=cv, n_jobs=-1, error_score='raise')

# report performance
print(f'nMAE: {mean(n_scores):.3f} ({std(n_scores):.3f})')

#%% bagging ensemble for making predictions for regression

# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15,
                       noise=0.1, random_state=5)

# define the model
model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [[0.88950817,-0.93540416,0.08392824,0.26438806,-0.52828711,-1.21102238,-0.4499934,1.47392391,-0.19737726,-0.22252503,0.02307668,0.26953276,0.03572757,-0.51606983,-0.39937452,1.8121736,-0.00775917,-0.02514283,-0.76089365,1.58692212]]
yhat = model.predict(row)
print(f'Prediction: {yhat[0]}')

#%% evaluate bagging algorithm for classification

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=5)

# define the model
model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv,
                           n_jobs=-1, error_score='raise')

# report performance
print(f'Accuracy: {mean(n_scores):.3f} ({std(n_scores):.3f})')

#%% make predictions using bagging for classification

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=5)

# define the model
model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [[-4.7705504,-1.88685058,-0.96057964,2.53850317,-6.5843005,3.45711663,-7.46225013,2.01338213,-0.45086384,-1.89314931,-2.90675203,-0.21214568,-0.9623956,3.93862591,0.06276375,0.33964269,4.0835676,1.31423977,-2.17983117,3.1047287]]
yhat = model.predict(row)
print(f'Predicted Class: {yhat[0]}')

#%%

"""
=============================================================================
Stacking
=============================================================================

Stacking ist eine Technik, die die Vorhersagen von mehreren Modellen (und die
Originaldaten, bei passthrough=True) als Input für einen Meta Learner benutzt.
Zweiter Input besteht aus dem Datensatz und den Vorhersagen der ersten Schicht.

Klassifikation und Regression
"""

im = img.imread('09_Ensemble_02.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()

#%%

im = img.imread('09_Ensemble_03.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()

#%% Code

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)
 
#%% Modelle

# Create Base Learners
base_learners = [
                 ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                 ('knn', KNeighborsClassifier(n_neighbors=5))             
                ]

# Initialize Stacking Classifier with the Meta Learner
clf = StackingClassifier(estimators=base_learners, 
                         final_estimator=LogisticRegression())

# Obwohl eine LogisticRegression, kann sie auch Klassen vorhersagen:
# (StackingClassifier)
 
# Extract score
print(clf.fit(X_train, y_train).score(X_test, y_test))

#%% Cross Validation

"""
Ein Problem beim Stacking ist, dass nicht klar ist, wo man die CV einbauen soll:
    Nur beim Meta Learner -> Overfitting, da die Base Learner schon overfittet
    haben könnten.

In Scikit-Learn wird es folgendermaßen implementiert:
    Base Learner sind auf dem vollen X gefittet, der final Estimator wird
    mit cv Vorhersagen der Base Learner trainiert.


Cross-validation default auf 5-fold, kann aber angepasst werden
"""

clf = StackingClassifier(estimators=base_learners,
                         final_estimator=LogisticRegression(),  
                         cv=10)

print(clf.fit(X_train, y_train).score(X_test, y_test))

#%%

"""
But that’s not all, you can also put in any cross-validation strategy you want:
"""

loo = LeaveOneOut()
clf = StackingClassifier(estimators=base_learners,
                         final_estimator=LogisticRegression(),
                         cv=loo)

print(clf.fit(X_train, y_train).score(X_test, y_test))

#%% Multi-layer Stacking

"""
Man kann auch mehrere Schichten von Base Learnern verwirklichen:
"""

im = img.imread('09_Ensemble_04.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()

#%%

"""
- Zwei Schichten von Base Learner:
    - Erste Schicht mit Random Forest und KNN
    - Zweite Schicht mit Decision Tree und Ramdom Forest
- Dann ein Meta Learner: Logistic Regression
"""

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Create Learners per layer
layer_one_estimators = [
                        ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
                        ('knn', KNeighborsClassifier(n_neighbors=5))             
                       ]
layer_two_estimators = [
                        ('dt', DecisionTreeClassifier()),
                        ('rf_2', RandomForestClassifier(n_estimators=50, random_state=42)),
                       ]

final_layer = StackingClassifier(estimators=layer_two_estimators, final_estimator=LogisticRegression())

# Create Final model by 
clf = StackingClassifier(estimators=layer_one_estimators, final_estimator=final_layer)

print(clf.fit(X_train, y_train).score(X_test, y_test))

#%% Stacking Ensemble Project (3. Quelle)

"""
Ensemble Stacking for Machine Learning and Deep Learning
"""

# Load and describe data
df = load_breast_cancer()
print(df.feature_names)
print(df.target_names)

#%%

X = pd.DataFrame(columns = df.feature_names, data = df.data)
y = df.target

print(X.head())
print(X.isnull().sum())
print(df.target.shape)

target = {'target' : df.target}
y = pd.DataFrame(data = target)

print(y.value_counts())

y = y['target']

print(X.describe())

#%% Define models

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
xgb = xgboost.XGBClassifier()

#%% K-fold cross Validation

clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)] # list of (str, estimator)

for name, algo in clf:
    score = cross_val_score(algo,X.values,y,cv = 5,scoring = 'accuracy')
    print(f"The accuracy score of {name} is:", score.mean())

#%% Stacking

dtc =  DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn =  KNeighborsClassifier()
xgb = xgboost.XGBClassifier()

clf = [('dtc',dtc),('rfc',rfc),('knn',knn),('xgb',xgb)] # list of (str, estimator)

lr = LogisticRegression()

stack_model = StackingClassifier(estimators = clf, final_estimator = lr)
score = cross_val_score(stack_model, X.values, y, cv = 5,
                        scoring = 'accuracy')
print("The accuracy score of the stacking ensemble is:", score.mean())
