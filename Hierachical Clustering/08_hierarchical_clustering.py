# -*- coding: utf-8 -*-
"""

"""

# %%Was ist Clustering

# Clustering ist eine unüberwachte Lerntechnik, bei der Daten in Gruppen 
# (Cluster) eingeteilt werden, die ähnliche Eigenschaften haben.

# Ziel: Die Identifikation von Mustern in den Daten ohne vorherige
# Beschriftung (Labels).

# Anwendungen: Marktforschung, Bildverarbeitung,
# Dokumentenklassifikation, etc.
# %%Was ist Hierarchische Clusteranalyse ?


# Hierarchische Clusteranalyse ist ein Verfahren, das Daten in eine 
# Hierarchie von Clustern unterteilt. Sie hat zwei Hauptmethoden:

# Agglomerative Hierarchische Clusteranalyse (bottom-up): Beginnt mit 
# jedem Punkt als eigenem Cluster und fusioniert dann schrittweise die 
# ähnlichsten Cluster.

# Divisive Hierarchische Clusteranalyse (top-down): Beginnt mit einem
#  einzigen Cluster, der alle Datenpunkte enthält, und teilt es 
#  schrittweise in kleinere Cluster.


# %%Verwendung von Scikit-Learn für Hierarchische Clusteranalyse

# In Scikit-Learn ist die Implementierung der hierarchischen 
# Clusteranalyse einfach und erfolgt über die Klasse
# AgglomerativeClustering

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

from skimage.data import coins  # Importiert das Münzbild aus skimage
from skimage.transform import rescale  # Für das Skalieren des Bildes
from skimage.filters import gaussian  # Für das Anwenden eines Gauss'schen Filters
from sklearn.feature_extraction.image import grid_to_graph  # Für die Konnektivitätsmatrix
import time  # Zum Messen der Laufzeit
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
import warnings
from itertools import cycle, islice
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler


# %%Beispiel-Daten
 
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Hierarchische Clusteranalyse anwenden
model = AgglomerativeClustering(n_clusters=2)  # Anzahl der Cluster
model.fit(X)

# Cluster Labels anzeigen
print("Cluster Labels:", model.labels_)

# Dendrogramm zeichnen
linked = linkage(X, 'single')  # 'single' ist der Linkage-Methoden-Parameter

plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.show()

# AgglomerativeClustering wird mit der Anzahl der gewünschten Cluster
# instanziiert.

# fit(X) führt die Clusteranalyse aus und labels_ gibt die zugewiesenen
# Cluster für jedes Datenpunkt zurück.

# linkage() aus dem scipy.cluster.hierarchy-Modul wird verwendet, um das
# Dendrogramm zu erstellen, das die hierarchische Struktur der Cluster
# darstellt.

# %%Präsentation und Visualisierung

# Um die Ergebnisse besser zu verstehen, nutzen wir das Dendrogramm
# Das Dendrogramm zeigt den Prozess der Clusterbildung,
# wobei jeder Schnitt des Baums die Fusion von zwei Clustern darstellt.

# Schritt 1: Iris-Datensatz laden
# Wir laden den Iris-Datensatz, der bereits in Scikit-Learn enthalten ist.
iris = load_iris()

# 'X' enthält die Merkmale (Länge und Breite der Kelch- und Blütenblätter).
X = iris.data  # Wir verwenden die 4 Merkmale: Sepal-Länge, Sepal-Breite, Petal-Länge, Petal-Breite

# Schritt 2: Hierarchische Clusteranalyse durchführen
# Wir verwenden die 'linkage'-Methode aus scipy, um die Clusteranalyse durchzuführen.
# Die Methode 'ward' minimiert die Varianz innerhalb der Cluster, was gut funktioniert,
# wenn wir versuchen, gleichmäßige Cluster zu erstellen.
linked = linkage(X, method='ward')  # Wir verwenden die 'ward'-Methode
# für die Clusterbildung

# Schritt 3: Dendrogramm erstellen
# Das Dendrogramm visualisiert die hierarchische Clusterstruktur. Es zeigt,
# wie die Punkte (Blumen) nach und nach zu Clustern zusammengefasst werden.

plt.figure(figsize=(10, 7))  
# Wir stellen sicher, dass das Diagramm eine angenehme Größe hat
dendrogram(linked)  # Zeichnet das Dendrogramm basierend auf den Clustern

# Schritt 4: Achsenbeschriftung und Titel hinzufügen
# Der Titel gibt an, dass es sich um die hierarchische Clusteranalyse 
#des Iris-Datensatzes handelt.
# Die X-Achse zeigt die Datenpunkte (Iris-Blumen) und die Y-Achse
# zeigt die Distanzen (Ähnlichkeit) bei der Clusterbildung.
plt.title("Dendrogramm der Hierarchischen Clusteranalyse (Iris-Datensatz)")  
# Titel des Diagramms
plt.xlabel("Index der Datenpunkte")  # Beschriftung der X-Achse
plt.ylabel("Distanzen")  # Beschriftung der Y-Achse

# Schritt 5: Plot anzeigen
# Hiermit wird das Dendrogramm angezeigt, das uns hilft, die Clusterbeziehungen 
#zu verstehen.
plt.show()  # Zeigt das Dendrogramm im Plot an


# %%Agglomerative Clustering mit verschiedenen Linkage-Strategien

# Schritt 1: Iris-Datensatz laden
iris = load_iris()
X = iris.data  # Wir verwenden die Merkmale des Iris-Datensatzes: Sepal und Petal Maße

# Schritt 2: Hierarchische Clusteranalyse durchführen und verschiedene 
#Linkage-Methoden vergleichen
# Hier werden verschiedene Linkage-Strategien getestet: 
    #'ward', 'complete', 'average', 'single'

linkage_methods = ['ward', 'complete', 'average', 'single']
for method in linkage_methods:
    # Anwendung der AgglomerativeClustering mit jeder Linkage-Methode
    agglomerative = AgglomerativeClustering(linkage=method)
    agglomerative.fit(X)

    # Dendrogramm erstellen (mit scipy linkage)
    # Zunächst müssen wir die Linkage-Matrix berechnen
    linked = linkage(X, method=method)

    # Schritt 3: Dendrogramm visualisieren
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title(f"Dendrogramm - Linkage Methode: {method}")
    plt.xlabel("Index der Datenpunkte")
    plt.ylabel("Distanz")
    plt.show()

# Für jede der Linkage-Methoden wird ein Dendrogramm erzeugt.
# Das Dendrogramm zeigt  die Hierarchie der Cluster. es kann 
# verwendet, um zu entscheiden, wie viele Cluster zu erstellen 
# . Das Dendrogramm wird so lange zusammengeführt, bis alle
# Datenpunkte in einem einzigen Cluster sind, und es kann durch die
# horizontale Linie entscheiden, wie viele Cluster extrahieren
# wollen.


# %%Connectivity Constraints
# Was sind Connectivity Constraints?
# Connectivity Constraints sind Einschränkungen, die den 
# AgglomerativeClustering-Algorithmus so verändern, dass nur 
# benachbarte Datenpunkte in einem Cluster zusammengeführt werden 
# dürfen



# %%Strukturierte Clusteranalyse auf einem Bild von Münzen

# Im folgenden Beispiel wenden wir den AgglomerativeClustering-Algorithmus
# auf ein Bild an, wobei Konnektivitätsbeschränkungen genutzt werden, um 
# benachbarte Pixel zusammenzufassen

# Schritt 1: Bilddaten laden
orig_coins = coins()  # Lädt das Standard-Münzbild aus skimage.data

# Schritt 2: Bild auf 20% der ursprünglichen Größe verkleinern, 
#um die Berechnungszeit zu verkürzen
smoothened_coins = gaussian(orig_coins, sigma=2) 
 # Ein Gauss'scher Filter wird auf das Bild angewendet, um Rauschen zu verringern
# Der Filter glättet das Bild und reduziert ungewollte Details.
# sigma=2 gibt die Standardabweichung des Filters an, um ein weiches
# Ergebnis zu erzielen.

rescaled_coins = rescale(
    smoothened_coins,  # Das geglättete Bild wird auf die neue Größe angewendet
    0.2,  # Verkleinerung des Bildes auf 20% der Originalgröße (schnellere Berechnungen)
    mode="reflect",  # Reflektiert die Bildränder beim Skalieren (verhindert Artefakte)
    anti_aliasing=False
    # Deaktiviert Anti-Aliasing, um die Verarbeitungszeit zu verkürzen
)

# Das Bild wird in ein 1D-Array umgeformt, damit es für den Clusteralgorithmus
# verwendet werden kann
X = np.reshape(rescaled_coins, (-1, 1)) 
 # Umformen in (Anzahl der Pixel, 1) für das Clustering

# Schritt 3: Struktur des Bildes definieren (Nachbarschaftsbeziehungen der Pixel)
# grid_to_graph erstellt eine Konnektivitätsmatrix für die Bildpixel, die nur
# benachbarte Pixel miteinander verbindet
connectivity = grid_to_graph(*rescaled_coins.shape)  
# Hier wird die Struktur des Bildes als Graph definiert

# Schritt 4: Hierarchische Clusteranalyse mit Ward-Linkage
n_clusters = 27  
# Anzahl der gewünschten Cluster (Regionen) im Bild
ward = AgglomerativeClustering(
    n_clusters=n_clusters,  
    # Anzahl der Cluster, in die das Bild unterteilt werden soll
    linkage="ward", 
    # Verwendung des Ward-Linkage-Ansatzes, der die Varianz innerhalb der Cluster 
    #minimiert
    connectivity=connectivity 
    # Die Konnektivitätsmatrix stellt sicher, dass nur benachbarte Pixel 
    #zusammengefasst werden
)

# Startzeit für das Clustering messen
print("Compute structured hierarchical clustering...")
start_time = time.time()  # Startzeit für die Berechnungen

# Durchführung der Clusteranalyse auf den Bilddaten
ward.fit(X)  # Führen Sie das Clustering mit den Bilddaten durch
label = np.reshape(ward.labels_, rescaled_coins.shape) 
 # Clusterlabels in die ursprüngliche Bildform umwandeln

# Laufzeit der Clusteranalyse anzeigen
print(f"Elapsed time: {time.time() - start_time:.3f}s")
  # Zeit, die für das Clustering benötigt wurde
print(f"Number of pixels: {label.size}")  # Anzahl der Pixel im Bild
print(f"Number of clusters: {np.unique(label).size}") 
 # Anzahl der gefundenen Cluster

# Schritt 5: Ergebnisse visualisieren
# Visualisierung der segmentierten Cluster auf dem Bild

plt.figure(figsize=(5, 5))  # Erstellen eines neuen Diagramms
plt.imshow(rescaled_coins, cmap=plt.cm.gray) 
 # Zeigt das Originalbild der Münzen im Graustufenmodus an

# Für jedes Cluster im Bild werden die Clustergrenzen gezeichnet
for l in range(n_clusters):
    # Contour wird verwendet, um die Grenzen jedes Clusters zu zeichnen
    plt.contour(
        label == l,  # Bedingung, die prüft, ob der Pixel zum Cluster 'l' gehört
        colors=[plt.cm.nipy_spectral(l / float(n_clusters))], 
        # Verwenden einer Farbskala für jedes Cluster
    )

# Deaktiviert die Achsen, um das Bild ohne Achsenanzeigen anzuzeigen
plt.axis("off")
plt.show()  # Zeigt das segmentierte Bild an

# %%Hierarchical Clustering: Structured vs Unstructured

# In dieser Analyse vergleichen wir zwei Arten des hierarchischen Clusterings auf 
# dem Swiss Roll Datensatz: unstrukturiertes und strukturiertes hierarchisches
# Clustering.

# Unstrukturiertes hierarchisches Clustering: Hierbei werden keine 
# Verbindungseinschränkungen angewendet.

# Strukturiertes hierarchisches Clustering: Hier wird zusätzlich ein 
# k-Nearest-Neighbors (k-NN) Graph verwendet, um die Clusterbildung mit einer 
# Verbindungseinschränkung zu versehen
# %%Beispiel Structured vs Unstructured

# 1. Swiss Roll generieren
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)

# Swiss Roll dünner machen (optional)
X[:, 1] *= 0.5

# 2. Unstrukturiertes hierarchisches Clustering (ohne Verbindungen)
print("Compute unstructured hierarchical clustering...")
start_time = time.time()
ward_unstructured = AgglomerativeClustering(n_clusters=6, linkage="ward").fit(X)
elapsed_time_unstructured = time.time() - start_time
labels_unstructured = ward_unstructured.labels_

print(f"Elapsed time (unstructured): {elapsed_time_unstructured:.2f}s")
print(f"Number of points: {labels_unstructured.size}")

# 3. Visualisierung des unstrukturierten Clusterings
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d", elev=7, azim=-80)
ax1.set_position([0, 0, 0.95, 1])

for l in np.unique(labels_unstructured):
    ax1.scatter(
        X[labels_unstructured == l, 0],
        X[labels_unstructured == l, 1],
        X[labels_unstructured == l, 2],
        color=plt.cm.jet(float(l) / np.max(labels_unstructured + 1)),
        s=20,
        edgecolor="k",
    )

fig1.suptitle(f"Unstructured Clustering (time {elapsed_time_unstructured:.2f}s)")
plt.show()  # Zeigt das Plot nur für unstrukturiertes Clustering

# 4. k-NN Graph für strukturiertes Clustering (Verbindungen erstellen)
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# 5. Strukturiertes hierarchisches Clustering (mit Verbindungen)
print("Compute structured hierarchical clustering...")
start_time = time.time()
ward_structured = AgglomerativeClustering(
    n_clusters=6, connectivity=connectivity, linkage="ward"
).fit(X)
elapsed_time_structured = time.time() - start_time
labels_structured = ward_structured.labels_

print(f"Elapsed time (structured): {elapsed_time_structured:.2f}s")
print(f"Number of points: {labels_structured.size}")

# 6. Visualisierung des strukturierten Clusterings
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="3d", elev=7, azim=-80)
ax2.set_position([0, 0, 0.95, 1])

for l in np.unique(labels_structured):
    ax2.scatter(
        X[labels_structured == l, 0],
        X[labels_structured == l, 1],
        X[labels_structured == l, 2],
        color=plt.cm.jet(float(l) / np.max(labels_structured + 1)),
        s=20,
        edgecolor="k",
    )

fig2.suptitle(f"Structured Clustering (time {elapsed_time_structured:.2f}s)")
plt.show()  # Zeigt das Plot nur für strukturiertes Clustering

# %%verschiedene hierarchische Verknüpfungsmethoden

# Erstellt verschiedene Toy-Datensätze (wie noisy_circles, noisy_moons, blobs, etc.), 
# die interessante Strukturen und Herausforderungen für Clusteralgorithmen darstellen.

# Verwendet vier verschiedene Verknüpfungsmethoden für das hierarchische Clustering:

# Ward
# Complete
# Average
# Single

# Jeder Clusteralgorithmus wird auf den Datensatz angewendet, und das Ergebnis 
# wird für jede Kombination von Datensatz und Verknüpfungsmethode in einem Plot
#  visualisiert.

# Der Plot wird mit Subplots organisiert, wobei jede Methode und jeder Datensatz 
# in einer eigenen Zelle des Plots angezeigt wird.


# Generate datasets
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=170)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=170)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=170)
rng = np.random.RandomState(170)
no_structure = rng.rand(n_samples, 2), None

# Anisotropically distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# Blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)

# Set up plot parameters
plt.figure(figsize=(9 * 1.3 + 2, 14.5))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01)

plot_num = 1

default_base = {"n_neighbors": 10, "n_clusters": 3}

datasets_list = [
    (noisy_circles, {"n_clusters": 2}),
    (noisy_moons, {"n_clusters": 2}),
    (varied, {"n_neighbors": 2}),
    (aniso, {"n_neighbors": 2}),
    (blobs, {}),
    (no_structure, {}),
]

# Loop through the datasets and apply the clustering algorithms
for i_dataset, (dataset, algo_params) in enumerate(datasets_list):
    # Update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # Normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # Create cluster objects for different linkage methods
    ward = cluster.AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="ward")
    complete = cluster.AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="complete")
    average = cluster.AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="average")
    single = cluster.AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="single")

    clustering_algorithms = (
        ("Single Linkage", single),
        ("Average Linkage", average),
        ("Complete Linkage", complete),
        ("Ward Linkage", ward),
    )

    # Loop through each clustering algorithm and plot the results
    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # Catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        # Create the subplot for each dataset and algorithm
        plt.subplot(len(datasets_list), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3",
                            "#999999", "#e41a1c", "#dede00"
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99, 0.01, ("%.2fs" % (t1 - t0)).lstrip("0"), transform=plt.gca().transAxes, size=15,
            horizontalalignment="right",
        )
        plot_num += 1

# Show the final plot with all the subplots
plt.show()

# %%Fazit
# Hierarchisches Clustering ist nützlich für die hierarchische Struktur 
# von Clustern und wenn keine Vorabdefinition der Anzahl der Cluster 
# notwendig ist.
















