# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:13:07 2025

@author: H. A.
"""
#************************ Hyperparameter Tuning***************************
"""
https://machinelearningmastery.com/hyperparameteroptimization-with-random-search-and-grid-search/
https://www.analyticsvidhya.com/blog/2020/09/alternativehyperparameter-optimization-technique-you-need-to-knowhyperopt/

Die Hyperparameter-Optimierung gilt als der schwierigste Teil beim Erstellen von 
Modellen für maschinelles Lernen und kann die Leistung eines Algorithmus deutlich steigern.

Hyperparameter sind Einstellungen, die vom Entwickler festgelegt werden, um den 
Lernprozess eines Modells für einen bestimmten Datensatz zu steuern.

Es gibt Hyperparameter (müssen manuell eingestellt werden) und 
Parameter: Werte, die das Modell während des Trainings aus den Daten lernt. 
                        (z. B. die Gewichte eines neuronalen Netzes).

Diese beiden Begriffe unterscheiden sich also: Parameter werden gelernt, während 
Hyperparameter manuell eingestellt werden müssen.
    
#----------------------------------------------

Hyperparameter-Optimierung mit Scikit-Learn

Beim Trainieren von Machine-Learning-Modellen gibt es Hyperparameter, die vor dem Training 
festgelegt werden müssen. Aber man weiß nicht immer, welche Werte die besten sind.

Scikit-Learn bietet zwei Methoden zur Optimierung dieser Hyperparameter:

    Random Search (Zufallssuche):

        Wählt zufällige Kombinationen aus dem definierten Suchraum aus.

        Ist oft effizienter als Grid Search, weil nicht jede Möglichkeit durchprobiert werden muss.
        Der Nachteil ist, dass wichtige Punkte (Werte) im Suchraum übersehen werden können.

    Grid Search (Raster-Suche):

        Probiert alle möglichen Kombinationen von Hyperparametern innerhalb eines definierten Gitters aus.

        Ist systematisch, aber kann sehr langsam sein, wenn es viele Kombinationen gibt.

Beide Methoden nutzen Cross-Validation (CV), um sicherzustellen, dass die gefundenen 
    Parameter auch wirklich gut generalisieren.


Zusammenfassung

    Hyperparameter-Tuning ist entscheidend, um das Beste aus einem Modell herauszuholen.
     
    Grid Search testet systematisch alle Möglichkeiten, ist aber langsam.

    Random Search probiert zufällige Kombinationen und ist oft effizienter.
    
    Ziel ist es, die beste Kombination von Hyperparametern zu finden.

"""
# %%
"""
Ziel der folgenden Beispiele ist es, eine leistungsfähige Modellkonfiguration für 
die Datensätze sonar (Klassifikation) und auto-insurance (Regression) zu finden.
"""

# %%

from pandas import read_csv  # Zum Laden des Datensatzes
from scipy.stats import loguniform  # Für die zufällige Auswahl von "C" aus einer logarithmischen Verteilung
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV  # Für Kreuzvalidierung und Hyperparameter-Suche
from sklearn.linear_model import LogisticRegression  # Das Modell für die Klassifikation

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"]="ignore"

# %% Zufallssuche (Random Search) für eine Klassifikationsaufgabe

# from pandas import read_csv  # Zum Laden des Datensatzes
# from scipy.stats import loguniform  # Für die zufällige Auswahl von "C" aus einer logarithmischen Verteilung
# from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV  # Für Kreuzvalidierung und Hyperparameter-Suche
# from sklearn.linear_model import LogisticRegression  # Das Modell für die Klassifikation

# Dataset laden
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'

# Laden des Datensatzes in ein DataFrame
dataframe = read_csv(url, header=None)

# Eingabe- und Ausgabewerte trennen
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# Modell definieren
model = LogisticRegression()

# (Evaluierungsstrategie definieren)
# Definieren der Kreuzvalidierungsstrategie:
# - 10-fache Kreuzvalidierung
# - 3 Wiederholungen zur besseren Stabilität der Bewertung
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Definieren des Hyperparameter-Suchraums
# leerer Dictionary-Bereich, in dem die verschiedenen Hyperparameter gespeichert werden, 
# die bei der Optimierung getestet werden.
space = dict()

# Der Solver bestimmt, welcher Algorithmus zur Optimierung der logistischen Regression genutzt wird.
# Die drei aufgelisteten Methoden sind:
# 'newton-cg': Ein optimierungsbasierter Solver, gut für größere Datensätze.
# 'lbfgs'    : Ein beliebter Solver, gut für große Datensätze und mehrdimensionale Probleme.
# 'liblinear': Besonders geeignet für kleine bis mittlere Datensätze und für L1-Regularisierung.
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']  # Verschiedene Optimierungsalgorithmen für die logistische Regression

space['C'] = loguniform(1e-5, 100)  # Der Regularisierungsparameter "C" wird aus einer 
                                    # logarithmischen Verteilung gezogen
space['fit_intercept'] = [True, False]  # Ob ein Bias-Term (Intercept) mittrainiert wird oder nicht


#  Zufallssuche einrichten
# (Erstellen des RandomizedSearchCV-Objekts:)
# - Modell: Logistische Regression
# - Suchraum: Definierte Hyperparameter
# - 500 Iterationen für die zufällige Suche
# - Bewertungsmaß: Genauigkeit (Accuracy)
# - Parallele Berechnung (n_jobs=-1 nutzt alle verfügbaren Prozessor-Kerne)
# - Kreuzvalidierung mit vorher definierter Strategie
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

# Hyperparameter-Suche durchführen (Training und Validierung)
result = search.fit(X, y)

# Beste gefundene Modellbewertung und Hyperparameter ausgeben
print('Beste Genauigkeit: %s' % result.best_score_)
print('Beste Hyperparameter: %s' % result.best_params_)

# %%
"""

Beste Hyperparameter:

    'C': 5.2221 → Der Regularisierungsparameter (C) bestimmt, wie stark das Modell reguliert wird. 
      Ein höherer Wert bedeutet weniger Regularisierung (mehr Freiheit für das Modell).

    'fit_intercept': True → Das Modell berücksichtigt einen Bias-Term (Intercept), was oft 
      sinnvoll ist, wenn die Daten nicht durch den Ursprung gehen.

    'solver': 'saga' → Der Optimierungsalgorithmus „saga“ wurde als bester gefunden. Er ist gut 
      für große Datensätze und funktioniert sowohl für L1- als auch für L2-Regularisierung.
    """

# %% Rastersuche (GridSearchCV) zur Klassifikation 

# from pandas import read_csv
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import GridSearchCV

# Datensatz laden
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = read_csv(url, header=None)
# Eingabe- und Ausgabewerte extrahieren
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# Modell definieren
model = LogisticRegression()

# Kreuzvalidierung definieren
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Suchraum für Hyperparameter definieren
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['l2']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

# Grid-Suche definieren
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)

# Suche ausführen
result = search.fit(X, y)

# Bestes Ergebnis ausgeben
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# %%  Zufallssuche (RandomizedSearchCV) zur Regression

# from scipy.stats import loguniform
# from pandas import read_csv
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import RandomizedSearchCV


# Datensatz laden
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
# Eingaben und Zielvariablen extrahieren
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# Modell definieren
model = Ridge()

# Kreuzvalidierung definieren
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Suchraum für Hyperparameter definieren
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]


# Zufallssuche konfigurieren
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)

# Suche ausführen
result = search.fit(X, y)

# Ergebnisse ausgeben
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# %%  Grid-Suche (GridSearchCV) zur Regression

# Grid-Suche für Ridge-Regression auf dem Auto Insurance Dataset
# from pandas import read_csv
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import GridSearchCV

# Datensatz laden
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)

# Daten in Eingabe- und Zielvariablen aufteilen
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# Modell definieren
model = Ridge()

# Evaluationsstrategie definieren (10-fache Kreuzvalidierung, 3 Wiederholungen)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Suchraum für Hyperparameter definieren
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]

# Grid-Suche definieren
search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

# Suche ausführen
result = search.fit(X, y)

# Beste Ergebnisse ausgeben
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# %%                    Alternative Hyperparameteroptimierung
"""
Es gibt verschiedene alternative Hyperparameter-Optimierungstechniken bzw. -methoden, 
um die besten Parameter für ein bestimmtes Modell zu erhalten.

    Hyperopt
    Scikit Optimize
    Optuna

Wir werden uns auf die Implementierung von Hyperopt konzentrieren.
"""

# %%  hyperopt

# Hyperopt ist eine leistungsstarke Python-Bibliothek zur Hyperparameteroptimierung
# Hyperopt verwendet eine Form der Bayesian-Optimierung für die Parametrierung
            # -> https://de.wikipedia.org/wiki/Bayes%E2%80%99sche_Optimierung
            
# %% 
   # Hyperopt enthält 4 wichtige Funktionen, die man kennen muss, um eine erste Optimierung durchzuführen.

# 1 Suchraum
# 2 Zielfunktion
# 3 Minimierungsfunktion fmin

from hyperopt import fmin, tpe, hp,Trials

trials = Trials()

best = fmin(fn=lambda x: x ** 2,
    		space= hp.uniform('x', -10, 10),
    		algo=tpe.suggest,
    		max_evals=50,
    		trials = trials)

print(best)

# 4 Trials Object (Versuchsobjekt)
# Das Trials-Objekt wird verwendet, um alle Hyperparameter, Verluste und andere Informationen zu speichern.
# Nach der Ausführung kann auf die einzelnen Schritte zugegriffen werden, um z.B. den Optimierungsprozess 
# fortzusetzen oder die Informationen anderweitig zu verwenden.

# %% Hyperopt in der Praxis (Mobile Price Dataset)
"""
Ziel ist die Entwicklung eines Modells zur Vorhersage des Preises eines Mobiltelefons 0 (niedrige Kosten) oder 1 (mittlere Kosten) oder 2 (hohe Kosten) oder 3 (sehr hohe Kosten).
"""

# import numpy as np 
# import pandas as pd 
# from sklearn.ensemble import RandomForestClassifier 
# from sklearn import metrics
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler 
# from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
# from hyperopt.pyll.base import scope

# import warnings
# warnings.filterwarnings("ignore")

# load data
data = pd.read_csv("mobile_price_data_train.csv")

#read data 
data.head()

#show shape
data.shape

#show list of columns 
list(data.columns)

#In unserem Datensatz sind verschiedene Merkmale mit numerischen Werten enthalten.

# %% Datensatz in Zielfunktionen und unabhängige Funktionen unterteilen.
#  Ziel ist Preis-Range.

# split data into features and target 
X = data.drop("price_range", axis=1).values 
y = data.price_range.values

# %% 4 Datensatz vorverarbeiten

# standardize the feature variables 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% 5 Parameterraum für Optimierung definieren
# Wir verwenden drei Hyperparameter des Random Forest Algorithmus, die n'estimators,max-depth und Kriterium sind.

space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    # "max_depth": hp.quniform("max_depth", 1, 15,1),
    "max_depth":    hp.choice('max_depth', np.arange(1, 16, dtype=int)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

# %% 6 Position definieren, um eine Minimierungsfunktion zu definieren (Objektivfunktion)

# define objective function

def hyperparameter_tuning(params):
    clf = RandomForestClassifier(**params,n_jobs=-1)
    acc = cross_val_score(clf, X_scaled, y,scoring="accuracy").mean()
    return {"loss": -acc, "status": STATUS_OK}

# %% 7 Fine Tune the Model

# Initialize trials object
trials = Trials()

best = fmin(
    fn=hyperparameter_tuning,
    space = space, 
    algo=tpe.suggest, 
    max_evals=100, 
    trials=trials
)

print("Best: {}".format(best))


# 100%|██████████| 100/100 [10:24<00:00,  6.25s/trial, best loss: -0.89]             
# Best: {'criterion': 1, 'max_depth': 12, 'n_estimators': 2}
 # n_estimators =  2 (number of trees)
    # → Es wurden nur 2 Entscheidungsbäume im Random Forest verwendet.

# %% 8: Analyse der Ergebnisse mit dem Versuchsobjekt

# trials.results
# Dies zeigt eine Liste von Wörterbüchern, die während der Suche durch "Adjektiv" 
                                                        # zurückgegeben wurden.
print(trials.results)
 
# trials.losses()
# Dies zeigt eine Liste von Verlusten (Schwamm für jede "ok"-Studie).
print(trials.losses())

# trials.statuses()
# Das zeigt eine Liste von Statusstrings.
print(trials.statuses())

# %%


'''
HyperOpt vs. HyperOpt-Sklearn

HyperOpt          ist eine leistungsstarke und vielseitige Bibliothek zur Hyperparameteroptimierung,
                  erfordert jedoch im Vergleich zu anderen Optionen mehr manuelle Konfiguration und Codierung
                 
HyperOpt-Sklearn  ist ein Wrapper, der auf HyperOpt aufbaut
                  und speziell auf Scikit-Learn-Modelle abzielt
                  vereinfacht die Konfiguration und Codierung


Vorteile:

Vereinfachte API:
HyperOpt-Sklearn verwendet eine Scikit-Learn-ähnliche API,
was die Anwendung erleichtert, wenn man mit der Bibliothek vertraut ist.

Automatische Suchraumdefinition:
Es extrahiert automatisch Hyperparameter aus Scikit-Learn-Modellen,
sodass keine manuelle Spezifikation erforderlich ist.

Integration mit Scikit-Learn-Pipelines:
Es lässt sich nahtlos in Scikit-Learn-Pipelines integrieren,
sodass Sie Hyperparameter innerhalb Ihres vorhandenen Workflows optimieren können.



Einschränkungen:

Geringere Flexibilität:
HyperOpt-Sklearn ist weniger flexibel als HyperOpt,
da es speziell für Scikit-Learn-Modelle entwickelt wurde.

Begrenzte Suchalgorithmen:
Es bietet im Vergleich zu HyperOpt einen kleineren Satz an Suchalgorithmen.



Verwendung von HyperOpt:
Wenn eine maximale Flexibilität und Kontrolle über den Optimierungsprozess benötigt wird
oder mit Nicht-Scikit-Learn-Modellen arbeiten


Verwendung von HyperOpt-Sklearn:
Wenn Scikit-Learn-Modelle verwendet werden
und eine einfachere, benutzerfreundlichere API bevorzugt wird


'''










