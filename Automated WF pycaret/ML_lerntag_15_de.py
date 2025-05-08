# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 14:07:28 2025

@author: Abdullah Kaplan
"""

# 1. Projektbeschreibung und Ziel
print("""
DIABETES-VORHERSAGE-PROJEKT
--------------------------
(https://www.analyticsvidhya.com/blog/2021/07/diabetes-prediction-with-pycaret/)
Ziel dieses Projekts ist es, mit PyCaret ein Machine-Learning-Modell 
zur Diabetes-Vorhersage zu entwickeln.
PyCaret ermöglicht den schnellen Vergleich mehrerer Klassifikatoren, 
um den besten Algorithmus auszuwählen.
""")

"""
# Was ist PyCaret?:
# PyCaret ist eine Python-Bibliothek, die den Machine-Learning-Prozess vereinfacht.
# Ähnlich wie die automatischen Bearbeitungsfunktionen eines Fotoeditors automatisiert 
PyCaret:
# - Datenvorbereitung
# - Modelltraining
# - Modellvergleich
# - Modelloptimierung
# Der Code wird auf ein Minimum reduziert.

# Beispiel: Sie können 15 verschiedene 
ML-Algorithmen mit einem einzigen Befehl vergleichen.
"""

##########################################
# 2. Notwendige Bibliotheken importieren
############################################


import pandas as pd  # Für Datenverarbeitung (ähnlich wie Excel-Tabellen)
import numpy as np   # Für mathematische Operationen (insbesondere numerische Berechnungen)
import matplotlib.pyplot as plt  # Grundlegende Diagramme (Liniendiagramme, Histogramme etc.)
import seaborn as sns  # Schönere und erweiterte Diagramme (baut auf Matplotlib auf)
from pycaret.classification import *  # Klassifikationsmodul von PyCaret
from pycaret.classification import plot_model  # Spezielle Funktion zur Visualisierung der Modellleistung
import warnings  # Für die Kontrolle von Warnmeldungen
warnings.filterwarnings('ignore')  # Technische Warnungen während der Ausführung werden ignoriert


#%%
############################################
# Warum diese Bibliotheken?
# - Pandas: Daten organisieren (Spalten hinzufügen/entfernen, filtern)
# - Numpy: Mathematische Operationen (Mittelwert, Standardabweichung)
# - Matplotlib/Seaborn: Visualisierung zum Verständnis der Daten
# - PyCaret: Vereinfacht Machine Learning
############################################

############################################
# 3. Datensatz laden und Spalten benennen
############################################

print("\nDatensatz wird heruntergeladen und geladen...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"  # Internetadresse der Daten (CSV-Format)

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']  # Spaltennamen, die wir definieren

data = pd.read_csv(url, names=columns)  # Lädt CSV von der URL und weist Spaltennamen zu

print("Datensatz erfolgreich geladen!")


#%%
############################################
# Details zum Laden der Daten:
# - pd.read_csv() lädt Daten von einer URL
# - names-Parameter: Weist Spaltennamen zu
# - data-Variable: Enthält alle Daten als DataFrame
############################################

############################################
# 4. Grundlegende EDA (Explorative Datenanalyse)
############################################

print("\nGRUNDLEGENDE DATENANALYSE")
print("========================")

print("\nGröße des Datensatzes:", data.shape)  # Zeigt (Anzahl_Zeilen, Anzahl_Spalten)

print("\nErste 5 Beobachtungen:")
print(data.head())  # Zeigt standardmäßig die ersten 5 Zeilen

print("\nGrundlegende Statistiken:")
print(data.describe())  # Zeigt Statistiken wie Count, Mean, Std, Min, Max für numerische Spalten

print("\nFehlende Werte:")
print(data.isnull().sum())  # Zeigt die Anzahl der fehlenden Werte (NaN) pro Spalte

############################################
# Was ist EDA?
# Erste Analysen zum Verständnis der Daten. Zum Beispiel:
# - Wie viele Patienten gibt es? (shape)
# - Wie sehen die Daten aus? (head)
# - Was ist der durchschnittliche Blutzucker? (describe)
# - Gibt es fehlende Daten? (isnull)
# Dieser Schritt ist entscheidend, um Probleme zu identifizieren.
############################################

############################################
# 5. Werte bereinigen
############################################

############################################
# Überprüfung auf 0-Werte (die nicht sinnvoll sind)
############################################

zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']  
# In diesen Spalten sind 0-Werte nicht sinnvoll

print("\nAnzahl der 0-Werte:")
print(data[zero_fields].eq(0).sum())  # Zählt die Anzahl der 0-Werte

#%%
"""
# Warum sind 0-Werte problematisch?
# - Glucose: 0 Blutzucker ist unmöglich (Messfehler)
# - BloodPressure: 0 Blutdruck bedeutet Tod
# - SkinThickness: 0 Hautdicke existiert nicht
# Diese Werte könnten fehlende Daten repräsentieren.
"""

############################################
# Ersetzen von 0-Werten durch NaN und Füllen mit dem Mittelwert
############################################

print("\n0-Werte werden korrigiert...")
for column in zero_fields:
    data[column] = data[column].replace(0, np.nan)  # Ersetzt 0 durch NaN
    mean_value = data[column].mean()  # Berechnet den Mittelwert der Spalte
    data[column] = data[column].fillna(mean_value)  # Füllt NaN mit dem Mittelwert
    print(f"0-Werte in Spalte {column} wurden durch Mittelwert {mean_value:.2f} ersetzt")

#%%
############################################
# Logik der Datenbereinigung:
# 1. Unlogische 0-Werte durch NaN ersetzen → Markiert die Daten
# 2. NaN-Werte mit Spaltenmittelwert füllen → Ersetzt durch plausible Werte
# Dieser Schritt ist entscheidend für korrektes Modelltraining.
############################################

############################################
# BMI-Ausreißer korrigieren
############################################

print("\nBMI-Ausreißer werden korrigiert...")
initial_outliers = len(data[data["BMI"]>40])  # Zählt Werte mit BMI > 40 (unrealistisch)

data["BMI"] = data["BMI"].apply(lambda x: data.BMI.mean() if x>40 else x)  
# Ersetzt Werte >40 durch Mittelwert

print(f"{initial_outliers} BMI-Ausreißer wurden durch Mittelwert ersetzt")

############################################
# Was sind Ausreißer?
# Werte, die weit außerhalb der normalen Verteilung liegen. Zum Beispiel:
# - Normaler BMI: 18-35
# - Werte über 40 könnten Messfehler oder Ausnahmen sein
# Diese Werte können das Modell irreführen und werden daher korrigiert.
############################################


#%%
############################################
# 6. Korrelationsmatrix
############################################

print("\nKorrelationsmatrix wird erstellt...")
plt.figure(figsize=(10,8))  # Diagrammgröße einstellen
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)  # Erstellt Heatmap
plt.title('Korrelationsmatrix zwischen Merkmalen')
plt.show()  # Zeigt das Diagramm

############################################
# Korrelationsanalyse:
# Werte zwischen -1 und +1:
# +1: Perfekte positive Korrelation (wenn einer steigt, steigt der andere)
# -1: Perfekte negative Korrelation (wenn einer steigt, fällt der andere)
# 0: Keine Korrelation
# Beispiel: Hohe Korrelation zwischen Glucose und Outcome zeigt Bedeutung für Diabetes-Vorhersage.
############################################


#%%
############################################
# 7. Verteilungsdiagramme
############################################

print("\nVariablenverteilungen werden visualisiert...")
fig, axs = plt.subplots(4, 2, figsize=(15,12))  # 4x2 Raster für Diagramme
axs = axs.flatten()  # Macht das Raster zu einer flachen Liste

# Verteilungsdiagramme für jede Spalte:
sns.distplot(data['Pregnancies'], rug=True, color='#38b000', ax=axs[0])
sns.distplot(data['Glucose'], rug=True, color='#FF9933', ax=axs[1])
sns.distplot(data['BloodPressure'], rug=True, color='#522500', ax=axs[2])
sns.distplot(data['SkinThickness'], rug=True, color='#66b3ff', ax=axs[3])
sns.distplot(data['Insulin'], rug=True, color='#FF6699', ax=axs[4])
sns.distplot(data['BMI'], color='#e76f51', rug=True, ax=axs[5])
sns.distplot(data['DiabetesPedigreeFunction'], color='#03045e', rug=True, ax=axs[6])
sns.distplot(data['Age'], rug=True, color='#333533', ax=axs[7])

plt.tight_layout()  # Verhindert überlappende Diagramme
plt.show()  # Zeigt alle Diagramme

############################################
# Warum Verteilungsdiagramme?
# - Hilft, die Datenstruktur zu verstehen
# - Ist die Verteilung normal (Glockenkurve)?
# - Gibt es Schiefe (skewness)?
# - Hilft, Ausreißer zu visualisieren
############################################



#%%
############################################
# 8. PyCaret-Setup
############################################

print("\nPyCaret-Klassifikationsumgebung wird eingerichtet...")
diab = setup(data = data, 
             target = 'Outcome',  # Zu vorhersagende Spalte
             session_id = 123,    # Fester Seed für Reproduzierbarkeit
             normalize = True,    # Skaliert Daten auf Bereich 0-1
             transformation = True,  # Korrigiert schiefe Verteilungen
             remove_multicollinearity = True,  # Entfernt hochkorrelierte Spalten
             multicollinearity_threshold = 0.9,  # Korrelationsschwellenwert
             fix_imbalance = True)  # Korrigiert Klassenungleichgewicht
print("PyCaret-Setup abgeschlossen!")

############################################
# PyCaret-Setup-Parameter:
# - target: Zielvariable für Vorhersage
# - session_id: Für reproduzierbare Ergebnisse
# - normalize: Skaliert alle numerischen Werte
# - transformation: Macht Daten normalverteilter
# - remove_multicollinearity: Entfernt redundante Merkmale
# - fix_imbalance: Korrigiert unausgeglichene Klassen
############################################



#%%
############################################
# 9. Vergleich aller Modelle
############################################

print("\nAlle Modelle werden verglichen...")
best_models = compare_models(sort='Accuracy', n_select=3)  # Sortiert nach Genauigkeit, wählt Top 3

print("\nBESTE 3 MODELLE:")
for i, model in enumerate(best_models, 1):
    print(f"{i}. Modell: {type(model).__name__}")  # Zeigt den Klassennamen des Modells

############################################
# Modellvergleich:
# - Accuracy: Prozentsatz korrekter Vorhersagen
# - AUC: Trennschärfe des Modells (je näher bei 1, desto besser)
# - Recall: Fähigkeit, Positive zu erkennen
# - Precision: Genauigkeit der positiven Vorhersagen
# - F1: Balance zwischen Precision und Recall
# - Kappa: Genauigkeit unter Berücksichtigung von Zufall
############################################



#%%
############################################
# 10. Beste Modelle erstellen und optimieren
############################################

print("\nBeste Modelle werden erstellt und optimiert...")
models = []  # Liste für optimierte Modelle

for model in best_models:
    model_name = type(model).__name__
    print(f"\n{model_name}-Modell wird erstellt...")
    created_model = create_model(model, fold=10)  # 10-fache Kreuzvalidierung
    print(f"{model_name}-Modell wird optimiert...")
    tuned_model = tune_model(created_model, optimize='Accuracy')  # Optimiert für Genauigkeit
    models.append(tuned_model)
    print(f"{model_name}-Modell erfolgreich erstellt und optimiert!")

############################################
# Was ist Hyperparameter-Optimierung?
# Automatische Anpassung der Modellparameter für beste Leistung.
# Beispiel für Random Forest:
# - Anzahl der Bäume
# - Tiefe
# - Stichprobenanteil
############################################

#%%
############################################
# 11. Spezielles Random-Forest-Modell erstellen und optimieren
############################################

print("\nZusätzliches Random-Forest-Modell wird erstellt...")
rf = create_model('rf', fold=10)  # 'rf' ist die Abkürzung für Random Forest
tuned_rf = tune_model(rf, optimize='Accuracy')
models.append(tuned_rf)
print("Random-Forest-Modell erfolgreich erstellt und optimiert!")

############################################
# Warum besonders Random Forest?
# - Gute Leistung in der Regel
# - Widerstandsfähig gegen Overfitting
# - Interpretierbar durch Feature Importance
############################################


#%%
############################################
# 12. Ensemble-Modell erstellen
############################################

print("\nTop-3-Modelle werden zu Ensemble-Modell kombiniert...")
blender = blend_models(estimator_list=models, method='soft')  # Kombiniert Modelle
print("Ensemble-Modell erfolgreich erstellt!")

############################################
# Was ist Ensemble Learning?
# Kombination mehrerer Modelle für bessere Vorhersagen.
# Ähnlich wie ein Ärztegremium, das gemeinsam entscheidet:
# - Jedes Modell "stimmt" ab
# - Die höchste Wahrscheinlichkeit gewinnt
############################################
#%%

############################################
# Finales Modell erstellen
############################################

print("\nFinales Modell wird erstellt...")
final_model = finalize_model(blender)  # Trainiert mit allen Daten
print("Finales Modell erfolgreich erstellt!")

############################################
# Warum finalize_model?
# - Vorher wurden Teile der Daten zum Testen zurückgehalten
# - Diese Funktion trainiert mit allen Daten
# - Macht das Modell produktionsbereit
############################################

#%%

############################################
# Modellbewertung
############################################
print("\nModellbewertungsmetriken:")
evaluate_model(final_model)  # Zeigt Leistungsmetriken

############################################
# Bewertungsmetriken:
# - Accuracy: Allgemeine Genauigkeit
# - AUC: Fähigkeit zur Unterscheidung der Klassen
# - Precision-Recall: Besser für unausgeglichene Datensätze
# - Confusion Matrix: Tatsächliche vs. vorhergesagte Klassen
# - Feature Importance: Welche Merkmale sind am wichtigsten
############################################
#%%
############################################
# 13. Visualisierungen
############################################

print("\nModellleistung wird visualisiert...")

print("\nROC-Kurve:")
plot_model(final_model, plot='auc')  # ROC: True Positive Rate vs. False Positive Rate

print("\nPrecision-Recall-Kurve:")
plot_model(final_model, plot='pr')  # Balance zwischen Precision und Recall

print("\nConfusion Matrix:")
plot_model(final_model, plot='confusion_matrix')  # Kreuztabelle der tatsächlichen und vorhergesagten Klassen

############################################
# Bedeutung der Diagramme:
# - ROC-Kurve: Je näher an der linken oberen Ecke, desto besser
# - Precision-Recall: Informativer für unausgeglichene Daten
# - Confusion Matrix:
#   - TP (True Positive): Korrekte positive Vorhersagen
#   - FP (False Positive): Falscher Alarm
#   - FN (False Negative): Übersehene Positive
############################################
#%%
############################################
# 14. Modellinterpretation
############################################

print("\nModellinterpretation wird durchgeführt (SHAP-Werte)...")
interpret_model(tuned_rf)  # Erklärt die Entscheidungsfindung des Modells

############################################
# Was sind SHAP-Werte?
# - Zeigt den Beitrag jedes Merkmals zur Vorhersage
# - Positive/negative Effekte werden farblich dargestellt
# - Macht das Modell weniger "blackbox"
############################################


#%%
############################################
# Modell speichern
############################################
print("\nModell wird gespeichert...")
save_model(final_model, 'diabetes_final_model_pycaret')  # Speichert Modell als Pickle-Datei

print("Modell wurde als 'diabetes_final_model_pycaret' gespeichert!")

############################################
# Pickle-Datei:
# - Speicherformat für Python-Objekte
# - Enthält alle Modellparameter und die Struktur
# - Kann später mit load_model() geladen werden
############################################


#%%
############################################
# Modell laden
############################################
print("\nModell wird geladen...")
loaded_model = load_model('diabetes_final_model_pycaret')  # Lädt gespeichertes Modell

print("Modell erfolgreich geladen!")

##########################################
# Modell laden:
# - Kein erneutes Training nötig
# - Direkt bereit für Vorhersagen
# - Kann in Webanwendungen oder APIs verwendet werden
##########################################
#%%
############################################
# 15. Vorhersage mit neuen Daten
############################################
print("\nVorhersage mit neuen Daten wird durchgeführt...")
new_data = pd.DataFrame({
    'Pregnancies': [2],          # 2 Schwangerschaften
    'Glucose': [120],            # 120 mg/dL Glukose
    'BloodPressure': [30],       # 70 mmHg Blutdruck
    'SkinThickness': [10],       # 10 mm Hautdicke
    'Insulin': [80],             # 80 μU/mL Insulin
    'BMI': [26.5],               # 26.5 kg/m² BMI
    'DiabetesPedigreeFunction': [0.25],  # Diabetes-Familiengeschichte
    'Age': [42]                  # 42 Jahre
})  # Patientendaten für Vorhersage

prediction = predict_model(loaded_model, data=new_data)  # Macht Vorhersage

print("\nVorhersagedetails:")
print(prediction)  # Zeigt Vorhersagelabel und -wahrscheinlichkeit

##########################################
# Vorhersageausgabe:
# - prediction_label: 0 (Negativ) oder 1 (Positiv)
# - prediction_score: Konfidenzniveau (0-1)
##########################################

# Vorhersageergebnis interpretieren
label = int(prediction['prediction_label'][0])  # Vorhersagelabel
score = float(prediction['prediction_score'][0])  # Vorhersagewahrscheinlichkeit

print("\nERGEBNIS:")
print("========")
if label == 1:
    print(f"Vorhersage: Diabetes POSITIV (Wahrscheinlichkeit: %{score*100:.2f})")
else:
    print(f"Vorhersage: Diabetes NEGATIV (Wahrscheinlichkeit: %{score*100:.2f})")

############################################
# Interpretation:
# - Über 70% wird meist als zuverlässig betrachtet
# - Ergebnisse nahe 50% können unsicher sein
# - Ärztliche Bestätigung ist erforderlich!
############################################

print("\nPROJEKT ERFOLGREICH ABGESCHLOSSEN!")