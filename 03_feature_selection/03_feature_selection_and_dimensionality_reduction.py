# -*- coding: utf-8 -*-
"""
@author: Swetlana
"""


# %%Featureauswahl mit Entscheidungsbäumen
import numpy as np 
 # Importiert NumPy für numerische Berechnungen.
import matplotlib.pyplot as plt 
 # Importiert Matplotlib für die Erstellung von Plots.
from sklearn.ensemble import RandomForestClassifier 
 # Importiert RandomForestClassifier erneut.
import seaborn as sns  # Importiert Seaborn für statistische Visualisierungen
 
 # Importiert Matplotlib für die Erstellung von Plots
import pandas as pd  # Importiert Pandas für die Arbeit mit DataFrames

from sklearn.decomposition import PCA
   # Importiert PCA für die Dimensionsreduktion.
from sklearn.preprocessing import StandardScaler 
  # Importiert StandardScaler zur Skalierung der Daten.
  
from sklearn.feature_selection import SelectKBest, chi2
  # Importiert SelectKBest für die Auswahl der besten Merkmale und chi2 für 
  #den Chi-Quadrat-Test.
from sklearn.preprocessing import LabelEncoder 
 # Importiert LabelEncoder, um kategoriale Daten in numerische zu konvertieren.
from sklearn.feature_selection import RFE 
  # Importiert RFE für die rekursive Merkmaleliminierung.
  

 
# %%Was ist Featureauswahl und Dimensionsreduktion ?

# Featureauswahl und Dimensionsreduktion sind entscheidende Techniken
# im maschinellen Lernen, um die Modellleistung zu verbessern und die 
# Berechnungsanforderungen zu reduzieren. Während die Featureauswahl 
# die wichtigsten Prädiktoren für ein Modell auswählt, transformiert 
# die Dimensionsreduktion die Daten, um eine kompaktere Darstellung 
# zu erhalten.
# %%Warum ist Featureauswahl wichtig?

# Vermeidung von Overfitting: Zu viele irrelevante Features können
# das Modell unnötig komplex machen.

# Verbesserung der Modellleistung: Relevante Features führen zu
# besseren Vorhersagen.

# Reduzierung der Berechnungskosten: Weniger Features 
# bedeuten schnellere Berechnungen und weniger Speicherbedarf.

# %%2. Methoden der Featureauswahl
# 2.1 Heuristische Methoden

# Diese Methoden basieren auf Expertenwissen, Korrelationen 
# oder visuellen Analysen.

# Beispiel: Korrelation zwischen Features



# Iris-Datensatz aus dem lokalen Pfad laden
iris = pd.read_csv('iris.csv')  
# Lade den Iris-Datensatz aus einer CSV-Datei. Sicherstellen,
# dass der Pfad korrekt ist.

# Nur numerische Spalten auswählen
iris_numeric = iris.select_dtypes(include=['float64', 'int64'])  
# Wählt nur die numerischen Spalten aus dem DataFrame.

# Korrelationsmatrix berechnen
corr_matrix = iris_numeric.corr() 
 # Berechnet die Korrelation zwischen den numerischen Merkmalen im Dataset.

# Heatmap der Korrelationen darstellen
plt.figure(figsize=(8,6))  # Definiert die Größe des Plots.
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
  # Zeigt die Korrelationsmatrix als Heatmap an.
plt.title("Korrelationsmatrix")  # Fügt dem Plot einen Titel hinzu.
plt.show()  # Zeigt den Plot an.

print(iris.head())

# Diese Matrix zeigt, wie stark Features miteinander korrelieren.
# Werte nahe +1 oder -1 deuten auf eine starke Korrelation hin

# %%2.2 Chi-Quadrat-Test (für kategoriale Daten)

# Der Chi-Quadrat-Test ist eine statistische Methode, um festzustellen,
#  ob zwei kategoriale Variablen unabhängig sind


# Features (X) und Zielvariable (y) trennen
X = iris.drop("Species", axis=1) 
 # Entfernt die Zielvariable "species" aus den Features.
y = LabelEncoder().fit_transform(iris["Species"]) 
 # Kodiert die Zielvariable "species" in numerische Werte.

# Featureauswahl mit Chi-Quadrat-Test
selector = SelectKBest(score_func=chi2, k=2) 
 # Wählt die zwei besten Features basierend auf dem Chi-Quadrat-Test aus.
X_new = selector.fit_transform(X, y) 
 # Wendet die Auswahl der besten Merkmale auf die Features und Zielvariable an.

# Ergebnisse anzeigen
print("Ausgewählte Features:", X.columns[selector.get_support()]) 
 # Gibt die Namen der ausgewählten Features aus.

# %%2.3 Recursive Feature Elimination (RFE)

# Recursive Feature Elimination (RFE) ist eine iterative Methode,
# die nach und nach die am wenigsten wichtigen Features entfernt.
# RFE für Klassifikation

# RFE arbeitet rekursiv, indem es weniger wichtige Features entfernt und das
# Modell erneut trainiert.

# zum Beispiel mit einem Entscheidungsbaum für Klassifikation:

# oder RFE für Regression

# RFE kann auch für Regressionsprobleme eingesetzt werden.




# Modell definieren
model = RandomForestClassifier(n_estimators=100, random_state=0) 
 # Erstellt ein Random Forest Modell mit 100 Bäumen.

# RFE durchführen
selector = RFE(estimator=model, n_features_to_select=2, step=1) 
 # Initialisiert den RFE-Selektor mit dem Modell, um 2 Features auszuwählen.
selector = selector.fit(X, y) 
 # Fittet den RFE-Selektor auf die Features und Zielvariable.

# Wichtige Features anzeigen
print("Wichtige Features:", X.columns[selector.support_])  
# Gibt die Namen der ausgewählten, wichtigen Features aus.



# Modell erstellen
model = RandomForestClassifier(n_estimators=100, random_state=0) 
 # Erstellt ein Random Forest Modell mit 100 Bäumen.
model.fit(X, y) 
 # Trainiert das Modell mit den Features und der Zielvariable.

# Feature-Importances berechnen
importances = model.feature_importances_  
# Berechnet die Wichtigkeit jedes Features basierend auf dem trainierten 
#Modell.
indices = np.argsort(importances)[::-1]  
# Sortiert die Features nach ihrer Wichtigkeit in absteigender Reihenfolge.

# Feature-Importances visualisieren
plt.figure(figsize=(8,6))  
# Definiert die Größe des Plots.
plt.title("Feature Importance") 
 # Fügt dem Plot einen Titel hinzu.
plt.bar(range(X.shape[1]), importances[indices], align="center") 
 # Erstellt einen Balkendiagramm für die Feature-Wichtigkeiten.
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45) 
 # Fügt die Feature-Namen als x-Achsen-Beschriftungen hinzu.
plt.show()  # Zeigt den Plot an.

#%%

#1. Modell definieren

model = RandomForestClassifier(n_estimators=100, random_state=0)

#%%

# Logistische Regression Beispiel
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("rfe", RFE(estimator=LogisticRegression(), n_features_to_select=1, step=1)),
    ]
)

pipe.fit(X, y)
ranking = pipe.named_steps["rfe"].ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)

# Add annotations for pixel numbers
for i in range(ranking.shape[0]):
    for j in range(ranking.shape[1]):
        plt.text(j, i, str(ranking[i, j]), ha="center", va="center", color="black")

plt.colorbar()
plt.title("Ranking of pixels with RFE\n(Logistic Regression)")
plt.show()

#%%
# Regression mit Crossvalidation

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1  # Minimum number of features to consider
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")

#%%

cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()

# %%Dimensionsreduktion
# Während die Featureauswahl bestimmte Variablen entfernt, 
# reduziert die Dimensionsreduktion die Anzahl der Dimensionen
# durch Transformationen.

# %% Hauptkomponentenanalyse (PCA)

# PCA ist eine Methode, um die Anzahl der Variablen in einem 
# Datensatz zu reduzieren, indem die Daten entlang der Hauptkomponenten
# (die größten Varianzen im Datensatz) projiziert werden. 
#  Dies führt zu einer Transformation der ursprünglichen Merkmale 
#  in eine neue, kleinere Anzahl von Merkmalen, die die meiste 
#  Information enthalten.



# Standardisierung der Daten
scaler = StandardScaler() 
 # Erstellt ein StandardScaler-Objekt, um die Daten zu standardisieren.
X_scaled = scaler.fit_transform(X)  
# Standardisiert die Features, um den Mittelwert 0 und die Standardabweichung 1 zu erreichen.

# PCA mit 2 Hauptkomponenten
pca = PCA(n_components=2) 
 # Initialisiert PCA, um die Daten auf 2 Hauptkomponenten zu reduzieren.
X_pca = pca.fit_transform(X_scaled) 
 # Führt PCA auf den standardisierten Daten aus.

# PCA-Ergebnisse visualisieren
plt.figure(figsize=(8,6))  # Definiert die Größe des Plots.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k') 
 # Erstellt einen Streudiagramm der 2 Hauptkomponenten.
plt.xlabel("Hauptkomponente 1") 
 # Beschriftet die x-Achse mit der ersten Hauptkomponente.
plt.ylabel("Hauptkomponente 2") 
 # Beschriftet die y-Achse mit der zweiten Hauptkomponente.
plt.title("PCA-Dimensionreduktion") 
 # Fügt dem Plot einen Titel hinzu.
plt.show()  # Zeigt den Plot an.

# %%Zusammenfassung 
# Featureauswahl hilft, irrelevante oder redundante Merkmale zu
# eliminieren, was zu einem effizienteren und besser generalisierenden
# Modell führt.
# Dimensionsreduktion wie PCA reduziert die Anzahl der Dimensionen,
# ohne wesentliche Informationen zu verlieren. Dies vereinfacht die Daten
# und hilft, Muster in hochdimensionalen Datensätzen besser zu verstehen.
# Beide Techniken tragen dazu bei, das Modell schneller zu machen,
# Overfitting zu vermeiden und die Modellinterpretation zu verbessern.
 
 
 
 
 
