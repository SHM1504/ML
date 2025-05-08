# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:23:06 2025

@author: H A


     ***Multi-Class-Classification mit dem Decision Tree aus ScikitLearn***

https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_t
ree_structure.html

Die Struktur eines Entscheidungsbaumes kann analysiert werden, um weitere
Einblicke in die Beziehung zwischen den Merkmalen (Features) und der
zu vorhersagenden Zielvariablen (Target) zu erhalten.

In diesem Beispiel wird gezeigt, wie die folgenden Informationen abgerufen werden können:
    1 die binäre Baumstruktur
    2 die Tiefe jedes Knotens und ob es sich um ein Blatt handelt
    3 die Knoten, die von einer Stichprobe (Sample) mit der decision_path-Methode erreicht wurden
    4 das Blatt, das von einer Stichprobe mit der apply-Methode erreicht wird
    5 die Regeln, die zur Vorhersage einer Stichprobe verwendet werden 
    6 der Entscheidungspfad, der von einer Gruppe von Stichproben gemeinsam genutzt wird

"""

import numpy as np
from matplotlib import pyplot as plt  # Für die Visualisierung des Entscheidungsbaums

from sklearn import tree  # Modul für Entscheidungsbäume
from sklearn.datasets import load_iris  # Laden des Iris-Datensatzes
from sklearn.model_selection import train_test_split  # Aufteilen der Daten in Trainings- und Testsets
from sklearn.tree import DecisionTreeClassifier  # Der Klassifikator für Entscheidungsbäume

# %% Train tree classifier

# Iris-Datensatz laden
iris = load_iris()
X = iris.data  # Merkmale (Features)
y = iris.target  # Zielvariablen (Klassen: 0, 1, 2)

# Aufteilen in Trainings- und Testdaten (75% Training, 25% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Erstellen eines Entscheidungsbaum-Klassifikators
clf = DecisionTreeClassifier(  # Initialisierung des Klassifikators
    max_leaf_nodes=3,  # Begrenzung auf maximal 3 Blattknoten, um die Baumgröße zu kontrollieren
    random_state=0  # Setzen des Zufallsseeds für Reproduzierbarkeit
)
    
clf.fit(X_train, y_train)  # Trainiert das Modell mit den Trainingsdaten

# %% 1. Binäre Baumstruktur abrufen (Analyse der Baumstruktur)

n_nodes = clf.tree_.node_count            # Anzahl der Knoten im Entscheidungsbaum
children_left = clf.tree_.children_left   # Linke Kindknoten
children_right = clf.tree_.children_right # Rechte Kindknoten
feature = clf.tree_.feature               # Welche Features für Splits verwendet werden
threshold = clf.tree_.threshold        # Schwellenwert für die Entscheidungsregel an jedem Knoten
values = clf.tree_.value                  # Klassenverteilungen in den Knoten

# %% 2. Tiefe jedes Knotens und ob es sich um ein Blatt handelt

# Erstellt ein Array mit der Anzahl der Knoten im Baum und setzt alle Werte auf 0
# Dieses Array speichert die Tiefe jedes Knotens im Baum.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)

# Erstellt ein weiteres Array mit der Anzahl der Knoten im Baum und setzt alle Werte auf False
# Dieses Array speichert, ob ein Knoten ein Blattknoten ist (True) oder nicht (False).
is_leaves = np.zeros(shape=n_nodes, dtype=bool)

# Initialisiert einen Stack mit dem Startknoten (Wurzelknoten ID = 0) und seiner Tiefe (0)
stack = [(0, 0)]

# Durchläuft den Entscheidungsbaum und bestimmt die Tiefe jedes Knotens sowie, ob er ein Blattknoten ist.
while len(stack) > 0:  # Solange noch Knoten im Stack sind...
    node_id, depth = stack.pop()  # Nimmt das letzte Element aus dem Stack (LIFO-Prinzip)
    node_depth[node_id] = depth  # Speichert die Tiefe des aktuellen Knotens

    # Prüft, ob der aktuelle Knoten ein Split-Knoten ist
    # Ein Split-Knoten hat unterschiedliche Werte für children_left und children_right
    is_split_node = children_left[node_id] != children_right[node_id]
    
    if is_split_node:
        # Falls es ein Split-Knoten ist, füge seine Kinder zum Stack hinzu.
        # Die Tiefe wird dabei um 1 erhöht.
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        # Falls der Knoten kein Split-Knoten ist, ist er ein Blattknoten
        is_leaves[node_id] = True

# Gibt die Baumstruktur aus
print("The binary tree structure has {n} nodes and has the following tree structure:\n".format(n=n_nodes))

#* Durchläuft alle Knoten und gibt entweder einen Blattknoten oder einen Split-Knoten aus.
for i in range(n_nodes):
    if is_leaves[i]:  #* Falls der Knoten ein Blattknoten ist
        print("{space}node={node} is a leaf node with value={value}.".format(
            space=node_depth[i] * "\t",  # Einrückung je nach Tiefe
            node=i,  # Knotennummer
            value=np.around(values[i], 3)  # Werte des Knotens gerundet
        ))
    else:  #* Falls der Knoten ein Split-Knoten ist
        print("{space}node={node} is a split node with value={value}: "
              "go to node {left} if X[:, {feature}] <= {threshold} "
              "else to node {right}.".format(
            space=node_depth[i] * "\t",  # Einrückung je nach Tiefe
            node=i,  # Knotennummer
            left=children_left[i],  # Linkes Kind
            feature=feature[i],  # Merkmal, das zum Splitten verwendet wurde
            threshold=threshold[i],  # Schwellenwert für die Entscheidung
            right=children_right[i],  # Rechtes Kind
            value=np.around(values[i], 3),  # Werte des Knotens gerundet
        ))

# %% Baum visualisieren

#* Zeichnet den Entscheidungsbaum grafisch mit matplotlib
tree.plot_tree(
    clf,        # Der trainierte Entscheidungsbaum-Klassifikator
    filled=True,  # Färbt die Knoten basierend auf den Klassen, um die Visualisierung zu verbessern
    proportion=True  # Skaliert die Knoten entsprechend der Anzahl der darin enthaltenen Datenpunkte
)
plt.show()

# Die Farbe repräsentiert die Klasse, der der Knoten angehört. 
# Je stärker die Farbe, desto sicherer ist sich der Knoten über seine Entscheidung.

# %% 3. Knoten, die von einer Stichprobe erreicht wurden (decision_path-Methode)

# Ermittelt den Entscheidungsweg für alle Testdaten
node_indicator = clf.decision_path(X_test)

# %% 4. Blattknoten, die von Stichproben erreicht werden (apply-Methode)

leaf_id = clf.apply(X_test)  # In welchem Blatt jeder Testpunkt landet

# %% 5. Regeln, die zur Vorhersage einer Stichprobe verwendet werden

sample_id = 0  # Wir analysieren die erste Testprobe

# IDs der Knoten, durch die die Testprobe geht
node_index = node_indicator.indices[ # `indices` enthält die Knoten-IDs
    node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    # `indptr` gibt die Positionen in `indices` an, die für `sample_id` relevant sind
]

# Gib die Regeln aus, die für das gewählte Testsample verwendet wurden
print("Rules used to predict sample {id}:\n".format(id=sample_id))

# Iteriere über alle Knoten, die das Sample durchläuft
for node_id in node_index:
    
    # Falls das aktuelle Blatt erreicht wurde, überspringen
    if leaf_id[sample_id] == node_id:
        continue # Wir brauchen die Blätter nicht zu analysieren, nur die Entscheidungsknoten

    # Prüfen, ob der Wert des Features kleiner oder größer als der Schwellenwert ist
    if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
        threshold_sign = "<="  # Falls ja, verwende das Zeichen `<=`
    else:
        threshold_sign = ">"   # Falls nein, verwende das Zeichen `>`

    # Gib die Entscheidungsregel für diesen Knoten aus
    print(
        "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
        "{inequality} {threshold})".format(
            node=node_id,  # Aktuelle Knoten-ID
            sample=sample_id,  # Test-Sample-ID
            feature=feature[node_id],  # Feature (Eingangsvariable), das im Knoten verwendet wird
            value=X_test[sample_id, feature[node_id]],  # Wert des Features für das Sample
            inequality=threshold_sign,  # Vergleichsoperator (`<=` oder `>`)
            threshold=threshold[node_id],  # Entscheidungsgrenze (Threshold) für den Knoten
        )
    )

# %% 6. Entscheidungspfad, der von einer Gruppe von Stichproben gemeinsam genutzt wird

# Wähle die Test-Samples aus, die verglichen werden sollen
sample_ids = [0, 1]  # Wir betrachten die Entscheidungspfade von Sample 0 und Sample 1

# Konvertiere die Sparse-Matrix `node_indicator` in eine dichte (normale) Matrix und 
# wähle nur die Zeilen aus, die zu den gewählten Samples gehören (sample_ids)
# `sum(axis=0)` summiert für jede Spalte (Knoten) die Anzahl der durchlaufenen Samples
common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
# common_nodes ist ein Boolean-Array, das angibt, welche Knoten von beiden Samples durchlaufen wurden.

# Erzeuge eine Liste mit den IDs der gemeinsamen Knoten
common_node_id = np.arange(n_nodes)[common_nodes]
# `np.arange(n_nodes)`: Erzeugt ein Array mit den Knoten-IDs von 0 bis `n_nodes - 1`
# `common_nodes` gibt für jede ID an, ob sie von beiden Samples besucht wurde (True/False)

# Gib die gemeinsamen Knoten aus
print(
      "\nThe following samples {samples} share the node(s) {nodes} in the tree.".format(
    samples=sample_ids, nodes=common_node_id
        )
    )
# Beispielausgabe: 
# The following samples [0, 1] share the node(s) [0 2] in the tree.

# Berechne und gib den Prozentsatz der gemeinsamen Knoten im Verhältnis zur Gesamtanzahl an Knoten aus
print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
# `len(common_node_id) / n_nodes` berechnet den Anteil der gemeinsam durchlaufenen Knoten
# `* 100` konvertiert den Wert in Prozent
# Beispielausgabe: 
# This is 40.0% of all nodes.

# Der Code analysiert, welche Knoten zwei Test-Samples gemeinsam durchlaufen haben.
# Das Ergebnis ist eine Liste gemeinsamer Knoten-IDs sowie ihr prozentualer Anteil an allen Knoten.

































