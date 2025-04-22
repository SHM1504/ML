# -*- coding: utf-8 -*-
"""
Created 22.04.2025

@author: S

Quelle: https://builtin.com/machine-learning/pca-in-python

"""



"""
# =============================================================================
# PCA - Principle Component Analysis
# =============================================================================


# ===== Das Prinzip =====

Was ist PCA?
    PCA ist eine Methode zur Datenreduktion:
    - Sie verwandelt viele Variablen (z. B. 100 Spalten) in wenige neue Variablen (z. B. 2 oder 3),
    - ohne dabei zu viele wichtige Informationen zu verlieren.

 Ziel: Komplexität reduzieren, aber die Struktur der Daten erhalten.

Warum brauchen wir das?
    - Viele Datensätze haben sehr viele Spalten (Features).
    - Manche Features sind stark miteinander korreliert (z. B. Körpergröße und Beinlänge).
    - Viel zu viele Dimensionen = schwer zu verarbeiten, schwer zu visualisieren.

PCA hilft:
    - Rechenzeit zu sparen  
    - Überflüssige Infos zu entfernen  
    - Daten als 2D- oder 3D-Plot darzustellen


Wie funktioniert PCA?

#  Schritt 1: Daten zentrieren / Skalieren?
    Alle Daten werden so verschoben, dass der Mittelwert jeder Spalte 0 ist.
    (StandardScaler: sorgt dafür, dass alle Merkmale denselben Wertebereich haben (Mittelwert = 0, Std = 1))

Warum? Für eine gute Vergleichbarkeit.


#  Schritt 2: Gemeinsamkeiten (Korrelation) finden
    PCA schaut:  
    - Welche Variablen sind ähnlich?  
    - Welche bewegen sich "zusammen"?

Beispiel:
| Körpergröße (cm) | Beinlänge (cm) |
||--|
| 180              | 95           |
| 170              | 90           |
| 160              | 85           |

 PCA erkennt: „Wer größer ist, hat auch längere Beine.“



#  Schritt 3: Neue Achsen berechnen (Hauptkomponenten)
    Statt „Körpergröße“ und „Beinlänge“ einzeln zu betrachten, erzeugt PCA neue Achsen:
    - 1. Hauptkomponente (PC1): Zeigt die Richtung der größten Streuung
    - 2. Hauptkomponente (PC2): Steht senkrecht zu PC1 und erklärt die zweitgrößte Streuung

Jede Hauptkomponente ist eine Kombination der Original-Features.

#  Schritt 4: Sortieren nach Wichtigkeit
PCA sortiert die neuen Achsen danach, wie viel der Gesamtinformation sie erklären:
- PC1 = z. B. 80% der Infos
- PC2 = 15%
- PC3 = 3%
- usw.

#  Schritt 5: Reduzieren
Du nimmst einfach nur die wichtigsten Komponenten (z. B. die ersten 2):
- Schon hast du viel weniger Daten
- Aber den Großteil der Info behalten!


  Konkretes Beispiel: Iris-Datensatz
Du hast 4 Merkmale:
- Kelchblattlänge
- Kelchblattbreite
- Blütenblattlänge
- Blütenblattbreite

PCA sagt dir:
- PC1 erklärt 72% der Varianz
- PC2 erklärt 23%
- PC3 + PC4 bringen fast nichts

 Du behältst PC1 und PC2 und kannst die Blütenarten jetzt in 2D darstellen.



  Wie sieht eine Hauptkomponente aus?
Beispiel-Formel für PC1:
```
PC1 = 0.5 * Kelchblattlänge + 0.3 * Kelchblattbreite + 0.6 * Blütenblattlänge + 0.4 * Blütenblattbreite
```
 PCA rechnet das für jede Zeile neu aus.



  Was bringt das in der Praxis?
| Ohne PCA                   | Mit PCA |

| 784 Pixel pro Bild (MNIST) | Nur 150 Hauptkomponenten |
| Lange Rechenzeit           | Schneller |
| Schwer zu visualisieren    | 2D-Plot möglich |



  Merksatz:
PCA dreht und kippt den Datenraum so, dass du die wichtigsten Unterschiede auf möglichst wenig Platz siehst
 – fast wie einen Apfel aus der besten Perspektive fotografieren.


  Noch einfacher: Metapher
Stell dir vor, du hast ein 3D-Objekt (z. B. ein Fußball).  
PCA projiziert ihn auf ein 2D-Bild (Foto), sodass möglichst viel Info erhalten bleibt.
"""

# %%


"""
===== Der Code =====
"""

#%% imports

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

# matplotlib, seaborn: für Visualisierungen
# numpy, pandas: Datenmanipulation
# sklearn: für PCA, Machine Learning und Preprocessing
# time: zur Messung der Laufzeit

#%% PCA Intuition

img = mpimg.imread("04_PCA/04_PCA/pca_intuition.webp")
plt.figure(figsize =(16,9), dpi=300)
plt.axis("off")
plt.title("PCA", fontsize = 32, loc="center", x=.5, y=.9)
plt.imshow(img)
plt.show()

# Zeigt ein Bild, das die Grundidee von PCA visualisiert (z.B. Datenprojektion).

#%% PCA Intuition II

img = mpimg.imread("04_PCA/04_PCA/pca_intuition_3.jpg")
plt.figure(figsize =(16,9), dpi=300)
plt.axis("off")
plt.title("PCA", fontsize = 32, loc="center", x=.5, y=.9)
plt.imshow(img)
plt.show()

#%% load dataset

penguins_in = sns.load_dataset("penguins")

# Holt den penguins-Datensatz von Seaborn. Dieser enthält biologische Merkmale dreier Pinguinarten.

#%% data inspection

print(penguins_in.head())
print(penguins_in.columns)
print(penguins_in.info())
print(penguins_in.describe())
print(penguins_in.isna().sum())

# Überblick verschaffen:
    # Spaltennamen
    # Datentypen
    # Fehlende Werte
    # Statistiken (Mittelwert, Standardabweichung etc.)

penguins_na = penguins_in[penguins_in.isna().any(axis=1)]
print(penguins_na)

# Zeilen mit fehlenden Werten anzeigen

#%% first plot

color_dict = {'Adelie': 'r', 'Chinstrap': 'g', 'Gentoo': 'b'}
# Convert the string values to their corresponding colors
colors = np.array(list(map(lambda x: color_dict[x], penguins_in.species)))
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=penguins_in.bill_length_mm, ys=penguins_in.flipper_length_mm,
           zs=penguins_in.body_mass_g,
           c=colors)

ax.set_xlabel('bill_length_mm')
ax.set_ylabel('flipper_length_mm')
ax.set_zlabel('body_mass_g')
plt.show()

# 3D-Streudiagramm der wichtigsten Merkmale - farblich nach Pinguinart.

#%% preprocessing: NaNs entfernen & Encoding

# drop missing values
penguins = penguins_in.dropna()

# encoding
le = LabelEncoder()
for column in penguins.columns:
    if penguins[column].dtype.name == "object":
        penguins.loc[:,column] = le.fit_transform(penguins[column])

# Kategorische Daten (z.B. species) in Zahlen umwandeln (Machine Learning braucht numerische Daten)

# check encoding
print(penguins.info())

# get feature and target names
features = list(penguins.columns)   # Holt sich alle Spaltennamen aus dem DataFrame penguins.
                                    # Speichert diese als Liste in der Variable features
targets = features.pop(0)           # pop(0) entfernt das erste Element der Liste features 
                                    # und speichert es in targets (erste Spalte enthält die Zielvariable (species))
print(features)
print(targets)

# species als Zielvariable (y)
# restliche Features als Input (X)

# get X and y dataframes
X = penguins[features]
y = penguins[targets]

# standardization
X = StandardScaler().fit_transform(X)

# Skalieren (StandardScaler): sorgt dafür, dass alle Merkmale denselben Wertebereich haben (Mittelwert = 0, Std = 1)

#%% PCA Projection to 2D

pca = PCA(n_components=2) # .95 for the amount of variance
pc = pca.fit_transform(X)

pcdf = pd.DataFrame(data = pc,
                    columns = ['principal component 1',
                               'principal component 2'])

print(pcdf)

# PCA reduziert die Datendimension auf 2 Hauptkomponenten.

# Dann wird das Ergebnis als 2D-Scatterplot dargestellt, farblich je Pinguin-Art.

color_dict = {0: 'r', 1: 'g', 2: 'b'}
# Convert the string values to their corresponding colors
colors = np.array(list(map(lambda x: color_dict[x], y)))
fig = plt.figure(dpi=300)
pcdf.plot.scatter(x='principal component 1',y='principal component 2',
                  c=colors, ax = plt.gca())
plt.show()

#%% Explained Variance
var_expl = pca.explained_variance_
var = pca.explained_variance_ratio_
print(var_expl)
print(var)

# Zeigt, wie viel Varianz jede der beiden Hauptkomponenten erklärt. (z.B. 70% + 20% = 90%)

#%%
"""
# =============================================================================
# PCA to Speed-Up Machine Learning Algorithms
# =============================================================================
"""

#%% Step 1: Download and load the Data

#  MNIST-Datensatz laden (Handgeschriebene Ziffern, 28x28 Pixel = 784 Features)

mnist = fetch_openml('mnist_784', parser='auto')

X = mnist.data
y = mnist.target

#%% Step 2: Split Data Into Training and Test Sets

# test_size: what proportion of original data is used for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 80% Trainingsdaten, 20% Testdaten

#%% Step 3: Standardize the Data

# Pixelwerte der Bilder normalisieren für bessere ML-Leistung
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)

# Apply transform to both the training set and the test set.
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Step 4: Train PCA

# PCA bestimmt, wie viele Komponenten nötig sind, um 95% der Varianz zu erhalten 

# Make an instance of the Model
pca = PCA(.95)

pca.fit(X_train_scaled)

#%% Step 5: Apply PCA

X_train_pc = pca.transform(X_train_scaled)
X_test_pc = pca.transform(X_test_scaled)

# Daten werden auf die berechneten Hauptkomponenten projiziert

#%% Step 6: Apply Logistic Regression to the Transformed Data

# all parameters not specified are set to their defaults
logreg = LogisticRegression(max_iter=1000, solver = 'lbfgs')

logreg.fit(X_train_pc, y_train)

# Klassifikationsmodell auf Basis der PCA-reduzierten Daten trainieren

#%% Step 7: Predict the labels of new data (new images).

# Predict for One Observation (image)
logreg.predict(X_test_pc[0:10])

#%% Step 8: Measuring Model Performance

score = logreg.score(X_test_pc, y_test)
print(score)

# Genauigkeit des Modells auf Testdaten ermitteln

#%% Step 9: Testing the Time to Fit Logistic Regression After PCA

# Zeitvergleich bei unterschiedlicher PCA-Varianz
def apply_pca(var):
    pca = PCA(var)
    X_train_pc = pca.fit_transform(X_train_scaled)
    X_test_pc = pca.transform(X_test_scaled)
    return X_train_pc, X_test_pc

taken_time = []

# Testet, wie sich die Rechenzeit und Genauigkeit verändern, wenn man mehr oder weniger Varianz behält.

for var in [.999,.99,.95,.9,.85]:
    X_train_pc, X_test_pc = apply_pca(var)
    start = time.time()
    logreg = LogisticRegression(max_iter=1000, solver = 'lbfgs')
    logreg.fit(X_train_pc, y_train)
    end = time.time()
    elapsed = end - start
    print(f"Time taken by Logistic Regression with PCA({var}):", elapsed)
    score = logreg.score(X_test_pc, y_test)
    print(f"Score: {score}")
