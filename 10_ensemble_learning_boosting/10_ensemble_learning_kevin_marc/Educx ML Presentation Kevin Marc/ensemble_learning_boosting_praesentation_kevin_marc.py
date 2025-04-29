#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 28 2025

@author: Kevin und Marc Ford

Sources: https://www.analyticsvidhya.com/blog/2021/04/best-boosting-algorithm-in-machine-learning-in-2021/
         https://medium.com/analytics-vidhya/catboost-101-fb2fdc3398f3
Dataset: https://www.kaggle.com/datasets/uciml/mushroom-classification?select=mushrooms.csv
"""
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

# Boosting ist eine Methode, die beim maschinellen Lernen verwendet wird, um die Genauigkeit
# eines Modells zu verbessern, indem die Vorhersagen mehrerer schwächerer Modelle kombiniert werden.
# Die Idee ist, eine Folge von Modellen stufenweise zu trainieren,
# wobei jedes Modell versucht, die Fehler des vorherigen Modells zu korrigieren.
# Die endgültige Vorhersage wird durch die Kombination der Vorhersagen aller
# Modelle in der Sequenz getroffen, in der Regel durch eine gewichtete Mehrheitsentscheidung.
# Einer der bekanntesten Boosting-Algorithmen heißt AdaBoost.


#%% Boosting Intuition

im = img.imread('boosting_intuition.png')
plt.figure(figsize=(16, 8))
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

""" AdaBoost 1995"""
input("Any Key for nex Cell (AdaBoost)")
# AdaBoost, kurz für Adaptive Boosting, ist ein Boosting-Algorithmus, der verwendet wird, um
# die Genauigkeit eines Modells zu verbessern, indem die Vorhersagen mehrerer
# schwächerer Modelle kombiniert werden. Es funktioniert, indem eine Folge von Modellen an die Daten angepasst wird,
# wobei jedes Modell versucht, die Fehler des vorherigen Modells zu korrigieren.
# Die endgültige Vorhersage wird durch die Kombination der Vorhersagen aller
# Modelle in der Sequenz gemacht, typischerweise durch eine gewichtete
# Mehrheitsentscheidung.


dtree = DecisionTreeClassifier()

cl = AdaBoostClassifier(n_estimators=100, estimator=dtree, learning_rate=1)
cl.fit(X_train,y_train)

y_pred = cl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of AdaBoost: ", accuracy)

#%%

""" Gradient Boosting 199/2001"""
input("Any Key for nex Cell (Gradient Boosting)")
# Gradient Boosting ist eine Methode zur Verbesserung der Genauigkeit eines Modells
# durch Training eines Ensembles von Entscheidungsbäumen. Es verwendet Gradientenabstieg
# um die Verlustfunktion zu minimieren, indem bei jeder Iteration ein neuer Baum angepasst wird,
# und dem Ensemble mit einem Gewicht hinzugefügt wird, das durch den aktuellen Verlust bestimmt wird.
# Die endgültige Vorhersage wird durch die Kombination der Vorhersagen aller
# Entscheidungsbäume im Ensemble erstellt.


# - We're creating an instance of the GradientBoostingClassifier
# - n_estimators=100: This specifies the number of decision trees that will be built in the
#   ensemble. More trees can sometimes lead to better performance but can also take longer
#   to train.
# - learning_rate=1.0: This controls the contribution of each tree to the final prediction.
#   A smaller learning rate might require more trees but can sometimes lead to a more robust
#   model.
# - max_depth=1: This limits the depth of each individual decision tree. A depth of 1 means
#   each tree will only make a decision based on a single feature. These are often called
#   "stumps."

cl = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)


cl.fit(X_train, y_train) # This is the training step. We're feeding our training
                         # features (X_train) and their corresponding target values (y_train)
                         # to the Gradient Boosting model so it can learn the patterns in the
                         # data.

y_pred = cl.predict(X_test)  # Once the model is trained, we use it to make predictions on the
                             # unseen test data (X_test). The predict() method outputs the
                             # model's guess for the class of each mushroom in our test set.
                             
# We then evaluate how well our Gradient Boosting model performed by comparing its
# predictions (y_pred) to the actual true classes in the test set (y_test).
# The accuracy_score calculates the percentage of predictions that were correct.
predictions = accuracy_score(y_test, y_pred)  

print("Accuracy of Gradient Boosting: ", predictions)

#%% Effect of the number of estimators (n_estimators) on the accuracy score

# Assuming X_train, y_train, X_test, y_test are already defined

n_estimators_range = range(10, 101, 10)
accuracy_scores = []

for n_estimators in n_estimators_range:
    cl = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1) 
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for n_estimators={n_estimators}: {accuracy}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracy_scores, marker='o')
plt.title('Gradient Boosting Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(n_estimators_range)
plt.show()

#%% Effect of the learning rate (learning_rate) on the accuracy score

# Assuming X_train, y_train, X_test, y_test are already defined
import numpy as np

learning_rate_range = np.arange(0.1, 1.5, 0.1)
accuracy_scores = []

for learning_rate in learning_rate_range:
    cl = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, max_depth=1)
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for learning_rate={learning_rate:.1f}: {accuracy}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(learning_rate_range, accuracy_scores, marker='o')
plt.title('Gradient Boosting Accuracy vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(learning_rate_range)
plt.show()

#%% Effect of the maximum depth (max_depth) of the individual trees on the accuracy score

# Assuming X_train, y_train, X_test, y_test are already defined

max_depth_range = range(1, 6)
accuracy_scores = []

for max_depth in max_depth_range:
    cl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=max_depth)
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for max_depth={max_depth}: {accuracy}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, accuracy_scores, marker='o')
plt.title('Gradient Boosting Accuracy vs. Maximum Depth')
plt.xlabel('Maximum Depth (max_depth)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(max_depth_range)
plt.show()

#%%

""" XGBoost - improvised version of the gradient boosting algorithm 2014/2016"""
input("Any Key for nex Cell (XGBoost)")
# XGBoost (eXtreme Gradient Boosting) ist eine Open-Source-Implementierung
# des Gradient-Boosting-Algorithmus. Er wurde entwickelt, um effizient und
# skalierbar zu sein, was ihn zu einer beliebten Wahl für große Aufgaben des maschinellen Lernens
# wie Klassifizierung und Regression macht. XGBoost ist bekannt für seine
# hohe Leistung, schnelle Trainingszeiten und die Fähigkeit, fehlende
# Werte und große Datensätze zu verarbeiten.

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=1, max_depth=1) # learning_rate=0.1, max_depth=3
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of XGBoost: ", accuracy)

#%% Effect of the number of estimators (n_estimators) on the accuracy score

# Assuming X_train, y_train, X_test, y_test are already defined

n_estimators_range = range(10, 101, 10)
accuracy_scores = []

for n_estimators in n_estimators_range:
    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=1, max_depth=1) 
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for n_estimators={n_estimators}: {accuracy}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracy_scores, marker='o')
plt.title('XGBoost Accuracy vs. Number of Estimators')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(n_estimators_range)
plt.show()

#%% Effect of the learning rate (learning_rate) on the accuracy score

import numpy as np

# Assuming X_train, y_train, X_test, y_test are already defined

learning_rate_range = np.arange(0.1, 1.6, 0.1)
accuracy_scores = []

for learning_rate in learning_rate_range:
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=learning_rate, max_depth=1) 
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for learning_rate={learning_rate:.1f}: {accuracy}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(learning_rate_range, accuracy_scores, marker='o')
plt.title('XGBoost Accuracy vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(learning_rate_range)
plt.show()

#%% Effect of the maximum depth (max_depth) of the individual trees on the accuracy score

# Assuming X_train, y_train, X_test, y_test are already defined

max_depth_range = range(1, 6)
accuracy_scores = []

for max_depth in max_depth_range:
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=max_depth)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Accuracy for max_depth={max_depth}: {accuracy}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, accuracy_scores, marker='o')
plt.title('XGBoost Accuracy vs. Maximum Depth')
plt.xlabel('Maximum Depth (max_depth)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(max_depth_range)
plt.show()

#%%

""" Catboost - Works well on Heterogeneous data. 2017+"""
input("Any Key for nex Cell (Catboost)")
# CatBoost ist ein Gradient-Boost-Algorithmus, der speziell für die
# Verarbeitung kategorischer Merkmale im Datensatz entwickelt wurde, ohne dass eine Vorverarbeitung
# erforderlich ist. Er verwendet eine neuartige Technik namens "permutationsbasierter Algorithmus"
# zur Behandlung kategorischer Variablen durch Berechnung der optimalen Split-Punkte auf
# dem kategorialen Merkmal anstelle der One-Hot-Codierungsmethode. Es verfügt außerdem über
# eine integrierte Behandlung fehlender Werte, ist effizient für große Datensätze
# und verfügt über integrierte Visualisierungstools, die das Verständnis des Modells erleichtern.

# Setze task_type zu CPU wenn du keine NVIDIA Grafikarte Hast (#GoodBoy)
# Ansonsten CPU (wenn MacUSer oder AMD User) (#LamePC)
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
    task_type = "GPU",
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


