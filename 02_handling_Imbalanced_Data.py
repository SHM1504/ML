# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:12:41 2025

@author: ad

            Handling Imbalanced Data
Quelle: https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
CSV: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Wenn die Beobachtung in einer Klasse (Class Imbalance) höher ist als in anderen Klassen,
liegt ein Klassenungleichgewicht vor.

Techniken von handling imbalanced data: Random undersampling, 
                                        Random oversampling, and NearMiss

- wenn imbalanced classes existieren, dann wird die Mehrheitsklasse vorhergesagt,
    und die Minderheitsklasse nicht erkannt.
Beispiel: ob die Kreditkartentransaktion betrügerisch war oder nicht in CSV erkennen

- Resampling Techniques to Solve Class Imbalance
    Dabei werden Stichproben aus der Mehrheitsklasse (majority class) entfernt
    (Under-Sampling) und/oder mehr Beispiele aus der Minderheitsklasse (minority class)
    hinzugefügt (Over-Sampling)
    
    Nachteile:
    * Over-Sampling führt zu overfishing
    * Under-Sampling führt zu Informationsverlust
    
"""
# %%

import pandas as pd
from IPython.display import display
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# import libraries imblearn under/over Sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

# load libraries sklearn
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# %%    Resampling Techniques to Solve Class Imbalance

im = Image.open("1_under_over-sampling.webp")
display(im)

# %% CSV einlesen

#%% load dataset

data = pd.read_csv(path + "/creditcard.csv")
data.columns = data.columns.str.strip()

#%% show distribution

print(data.head())

print(data.shape)

#%% create subset 9000 rows

# separate fraudulent and non fraudulent data
data_0 = data[data['Class'] == 0]
data_1 = data[data['Class'] == 1]

# take only 9000 0's sample
data_0 = data_0.sample(n=9000)

# combine both dataframes
data = pd.concat([data_1, data_0])

print(data.Class.value_counts())


# %% 

# class count
class_count_0, class_count_1 = data['Class'].value_counts()

# Separate class into majority and minority
class_0 = data[data['Class'] == 0]  # majority class
class_1 = data[data['Class'] == 1]  # minority class

# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

# %%    1. Random Under-Sampling
# Undersampling kann definiert werden als das Entfernen einiger Beobachtungen
# der majority class. Dies geschieht so lange, bis die majority and minority class
# ausgeglichen ist.

# Undersampling kann eine gute Wahl sein, wenn eine große Datenmenge vorhanden
# Ein Nachteil des Undersampling ist jedoch, dass wir Informationen entfernen,
# die möglicherweise wertvoll sind.

# Undersampling von majority_class durchführen
class_0_under = class_0.sample(class_count_1)

# Test_under Daten
test_under = pd.concat([class_0_under, class_1], axis=0)

# print the count after under-sampeling
print("total class of 1 and0:",test_under['Class'].value_counts())

# plot the count after under-sampeling
test_under['Class'].value_counts().plot(kind='bar', title='count (target)')

# %%        2. Random Over-Sampling
# Oversampling kann als Hinzufügen weiterer Kopien zur minority class
# definiert werden. Oversampling kann beim maschinellen Lernen eine gute Wahl
# sein, wenn Sie nicht mit einer großen Menge an Daten arbeiten können.

# Ein Nachteil des Oversamplings ist, dass es zu einer overfitting  und einer
# schlechten generalization auf den Testsatz führen kann

# Over-Sampling von minority class durchführen
class_1_over = class_1.sample(class_count_0, replace=True)

# Test_over Daten
test_over = pd.concat([class_1_over, class_0], axis=0)

# print the count after under-sampeling
print("total class of 1 and 0:",test_under['Class'].value_counts())

# plot the count after under-sampeling
test_over['Class'].value_counts().plot(kind='bar', title='count (target)')

# %%        Balance Data With the Imbalanced-Learn Python Module
# resampling techniques using the Python library imbalanced-learn

# %%        Random Under-Sampling With Imblearn

# X-Daten bis zu letzten Spalten
x = data.iloc[:, 0:-1].values

# y-Daten letzte Spalte
y = data.iloc[:, -1].values

rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
x_rus, y_rus = rus.fit_resample(x, y)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_rus))

# print('original dataset shape:', y.shape)
# print('Resample dataset shape', y_rus.shape)

# %%        Random Over-Sampling With imblearn

ros = RandomOverSampler(random_state=42)

# fit predictor and target variable
x_ros, y_ros = ros.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))

# print('original dataset shape:', y.shape)
# print('Resample dataset shape', y_ros.shape)

# %%        Under-Sampling: Tomek Links
# neue Stichproben durch Zufallsstichproben erzeugen, wobei die derzeit
# verfügbaren Stichproben ersetzt werden.

im = Image.open("1_tomek_links.webp")
display(im)

# %%

tl = TomekLinks(sampling_strategy='majority')

# fit predictor and target variable
x_tl, y_tl = tl.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_tl))

# %%        Synthetic Minority Oversampling Technique (SMOTE)

# SMOTE (Synthetic Minority Oversampling Technique im maschinellen Lernen)
# funktioniert durch die zufällige Auswahl eines Punktes aus der minority class
# und die Berechnung der k-nächsten Nachbarn für diesen Punkt. Die synthetischen
# Punkte werden zwischen dem ausgewählten Punkt und seinen Nachbarn hinzugefügt.

im = Image.open("1_smote.webp")
display(im)

# SMOTE algorithm works in 4 simple steps:
#   - a minority class as the input vector.
#   - finde k-neighbors (k_neighbors is specified as an argument in the SMOTE() function).
#   - platzieren einen synthetischen Punkt an einer beliebigen Stelle auf der Linie
#        zwischen dem betrachteten Punkt und seinem gewählten Nachbarn
#   - Wiederholen die Schritte, bis die Daten ausgeglichen sind.
# %%


smote = SMOTE()

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_smote))

# %%        NearMiss

# NearMiss ist eine Technik der under-sampling. Anstatt die minority class neu
# zu erfassen, wird die majority class durch die Verwendung eines Abstands mit
# der minority class gleichgesetzt.

nm = NearMiss()

x_nm, y_nm = nm.fit_resample(x, y)

print('Original dataset shape:', Counter(y))
print('Resample dataset shape:', Counter(y_nm))

# %%        Change the Performance Metric
# Die Genauigkeit ist nicht die beste Metrik für die Bewertung unausgewogener
# Datensätze, da sie irreführend sein kann.


# Metriken, die einen besseren Einblick geben können, sind:
    
# - Confusion Matrix

# - Precision: die Anzahl wahrer positiver Ergebnisse dividiert durch alle
#    positiven Vorhersagen. Die Präzision wird auch als positiver Vorhersagewert
#    bezeichnet. Sie ist ein Maß für die Genauigkeit eines Klassifikators.
#    Eine niedrige Präzision deutet auf eine hohe Anzahl von falsch-positiven
#    Vorhersagen hin.

# - Recall: die Anzahl der wahrhaft positiven Werte geteilt durch die Anzahl
#    der positiven Werte in den Testdaten. Der Recall wird auch als Sensitivität
#    oder True Positive Rate bezeichnet. Sie ist ein Maß für die Vollständigkeit
#    eines Klassifikators. Eine niedrige Rückrufquote deutet auf eine hohe Anzahl
#    falsch negativer Werte hin.  

# - F1-Score: der gewichtete Durchschnitt von Precision und Recall

# - Area Under ROC Curve (AUROC): Der AUROC-Wert gibt die Wahrscheinlichkeit an,
#     mit der Ihr Modell Beobachtungen aus zwei Klassen unterscheiden kann.
#     Mit anderen Worten: Wenn Sie zufällig eine Beobachtung aus jeder Klasse
#     auswählen, wie hoch ist dann die Wahrscheinlichkeit, dass Ihr Modell in
#     der Lage ist, sie korrekt zu „klassifizieren“?

# Berechnung von True Positive Rate (TPR) = TruePositive / (TruePositive + FalseNegative)
# Falsch-positiv-Rate (FPR) = FalsePositive / (FalsePositive + TrueNegative)

# TP oder True Positives sind die Anzahl der positiven Klassendatensätze,
# die das Modell korrekt als positiv vorhersagt.

# FN oder falsche Negative sind die Anzahl der positiven Klassendatensätze,
# die das Modell fälschlicherweise als negativ vorhersagt.

# FP oder False Positives sind die Anzahl der negativen Klassendatensätze,
#  die fälschlicherweise als positiv vorhergesagt werden.

# TN oder wahre Negative sind die Anzahl der negativen Klassendatensätze,
#  die korrekt als negativ vorhergesagt wurden.

im = Image.open("fig-multilabel_confusion_matrix.png")
display(im)

# %%        Penalize Algorithms (Cost-Sensitive Training)
# Die nächste Taktik besteht darin, bestrafte Lernalgorithmen zu verwenden,
# die die Kosten für Klassifizierungsfehler in der Minderheitenklasse erhöhen.

# Ein beliebter Algorithmus für diese Technik ist Penalized-SVM.

# Während des Trainings können wir das Argument class_weight='balanced' verwenden,
#  um Fehler in der Minderheitenklasse mit einem Betrag zu bestrafen, der
#  proportional zur Unterrepräsentation dieser Klasse ist.

# Wir wollen auch das Argument probability=True einbeziehen, wenn wir
#  Wahrscheinlichkeitsschätzungen für SVM-Algorithmen aktivieren wollen.


### Samples data für x_test, y_test organisieren
data_0_svc = data[data['Class'] == 0]
data_1_svc = data[data['Class'] == 1]

# take only 3000 0's sample
data_0_svc = data_0_svc.sample(n=3000)

# combine both dataframes
data_svc = pd.concat([data_1_svc, data_0_svc])

# X_test-Daten bis zu letzten Spalten
x_test = data_svc.iloc[:, 0:-1].values

# y_test-Daten letzte Spalte
y_test = data_svc.iloc[:, -1].values

# %%


# we can add class_weight='balanced' to add panalize mistake
svc_model = SVC(class_weight='balanced', probability=True)

# x, y sind von 9000 sample data
svc_model.fit(x, y)

svc_predict = svc_model.predict(x_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))

# %%        Plotten ROC Curve

# Predict the probabilities for the test set
y_probs = svc_model.predict_proba(x_test)[:, 1]

# Predict the classes for the test set
y_pred = svc_model.predict(x_test)

# Plotting the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %%        Change the Algorithm

# Entscheidungsbäume schneiden bei unausgewogenen Daten häufig gut ab.
# Beim modernen maschinellen Lernen übertreffen Baum-Ensembles
# (Random Forests, Gradient Boosted Trees usw.) fast immer einzelne Entscheidungsbäume

rfc = RandomForestClassifier()

# fit the predictor and target
rfc.fit(x, y)

# predict
rfc_predict = rfc.predict(x_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, rfc_predict))
print('Accuracy score:',accuracy_score(y_test, rfc_predict))
print('F1 score:',f1_score(y_test, rfc_predict))