# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 2025

@author: ad

Quelle: https://scikit-learn.org/stable/modules/clustering.html#dbscan
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html
https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html



             *** Clustering-Verfahren (K-Means + DBSCAN) ***
Clustering von unlabeled Daten kann mit dem Modul sklearn.cluster durchgeführt werden.

Für jeden Clustering-Algorithmus gibt es zwei Varianten: 
  - eine Klasse, die die Fit-Methode implementiert, um die Cluster auf den Trainingsdaten zu lernen, 

  - und eine Funktion, die bei gegebenen Trainingsdaten ein Array von ganzzahligen
  Labels zurückgibt, die den verschiedenen Clustern entsprechen.
  
Für die Klasse können die Labels über die Trainingsdaten im Attribut labels_ gefunden werden.

===========================
Input data

Ein wichtiger Punkt ist, dass die in diesem Modul implementierten Algorithmen
verschiedene Arten von Matrizen als Eingabe akzeptieren können. Alle Methoden
akzeptieren Standard-Datenmatrizen der Form (n_samples, n_features).
Diese können von den Klassen im Modul sklearn.feature_extraction bezogen werden.

Für AffinityPropagation, SpectralClustering und DBSCAN kann man auch Ähnlichkeitsmatrizen
der Form (n_samples, n_samples) eingeben. Diese können mit den Funktionen des Moduls
sklearn.metrics.pairwise ermittelt werden.

============================

Method name:    K-Means
------------------------
Parameters:     Anzahl von clusters          
Scalability:    große n_samples, mittlere n_clusters mit MiniBatch code  
Usecase:        Universell einsetzbar, gleichmäßige Clustergröße, flache Geometrie,
                nicht zu viele Cluster, induktiv
Geometry (metric used): Distances between points


Method name:    DBSCAN (density-based spatial clustering of application with noise)
------------------------
Parameters:     neighborhood size          
Scalability:    Very large n_samples, medium n_clusters  
Usecase:        Non-flat geometry, ungleiche cluster sizes, outlier removal, transductive
Geometry (metric used): Abstände zwischen den nächstgelegenen Punkten

** Das Clustering mit nicht flacher Geometrie ist nützlich, wenn die Cluster eine
 bestimmte Form haben, d. h. eine nicht flache Mannigfaltigkeit, und der euklidische
 Standardabstand nicht die richtige Metrik ist. 

** KMeans kann als ein Spezialfall eines Gaußschen Mischungsmodells mit gleicher
 Kovarianz pro Komponente betrachtet werden.

** Transduktive Clustering-Methoden (im Gegensatz zu induktiven Clustering-Methoden)
 sind nicht dafür ausgelegt, auf neue, ungesehene Daten angewendet zu werden.
 
 
"""
# %%

from IPython.display import display
from PIL import Image

im = Image.open("sphx_glr_plot_cluster_comparison_001.png")
display(im)

# %%        Der KMeans-Algorithmus

"""
* Der k-means-Algorithmus unterteilt eine Menge von Proben in disjunkte Cluster,
 die jeweils durch den Mittelwert der Proben im Cluster beschrieben werden.
 Die Mittelwerte werden gemeinhin als Cluster-„Zentren (centroid)“ bezeichnet; dabei ist
 zu beachten, dass es sich im Allgemeinen nicht um Punkte handelt, auch wenn sie
 sich im selben Raum befinden.

Der K-means-Algorithmus zielt darauf ab, centroids zu wählen, die das Kriterium
 der Trägheit bzw. der Quadratsumme innerhalb des Clusters minimieren:
     
Nachteile:
------------
* Es reagiert schlecht auf langgestreckte Cluster oder Verteiler mit unregelmäßigen Formen.
* Die Trägheit (inertia) ist keine normalisierte Metrik: .. niedrigere Werte besser und Null optimal ist.     
    Die Durchführung eines Algorithmus zur Dimensionalitätsreduzierung, wie z. B.
    der Hauptkomponentenanalyse (PCA), vor dem k-means-Clustering kann ... die Berechnungen beschleunigen.
    
    
* es gibt ein Initialisierungsschema k-means++

* Der Algorithmus unterstützt Stichprobengewichte, die durch einen Parameter 
sample_weight angegeben werden können. Dadurch kann einigen Stichproben bei 
der Berechnung von Clusterzentren und Trägheitswerten ein höheres Gewicht zugewiesen werden.

"""   

im = Image.open("sphx_glr_plot_kmeans_assumptions_002.png")
display(im)
  
# %%    Dieses Beispiel soll Situationen veranschaulichen, in denen k-means 
# unintuitive und möglicherweise unerwünschte Cluster erzeugt.

import numpy as np

# Die Funktion make_blobs erzeugt isotrope (sphärische) Gauß-Blobs.
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170

# Um Gauß-Blobs zu erhalten, muss man eine lineare Transformation definieren.
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

# Daten in X, y aufteilen mit make_blobs()
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# X-Daten transformieren in Array[1500,2] => Anisotropic blobs
# np.dot: Dot product of two arrays
X_aniso = np.dot(X, transformation)  

# Für X- und y-Daten Gauß-Blobs erzeugen mit make_blobs => Ungleiche Varianzen 
X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)  # Unequal variance

# X- und y-Daten filtern => Ungleichmäßig große Blobs
# np.vstack: Stack arrays in sequence vertically (row wise).
X_filtered = np.vstack(
    (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
)  

# wenn y-value = 0, dann 500 rows, y=1 dann 100 , y=2 dann 10 rows; in der Summe 610 rows

y_filtered = [0] * 500 + [1] * 100 + [2] * 10

# %%    visualize the resulting data

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

# sample X-daten "Mixture of Gaussian Blobs" plotten
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y)
axs[0, 0].set_title("Mixture of Gaussian Blobs")

# X_aniso Daten "Anisotropically Distributed Blobs" plotten
axs[0, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y)
axs[0, 1].set_title("Anisotropically Distributed Blobs")

# X_varied "Unequal Variance" plotten
axs[1, 0].scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied)
axs[1, 0].set_title("Unequal Variance")

# X_filtered "Unevenly Sized Blobs" plotten
axs[1, 1].scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered)
axs[1, 1].set_title("Unevenly Sized Blobs")

plt.suptitle("Ground truth clusters").set_y(0.95)
plt.show()

# %%

from sklearn.cluster import KMeans

common_params = {
    "n_init": "auto",
    "random_state": random_state,
}

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

# Predictions von X-daten erstellen mit KMeans, n_clusters=2
y_pred = KMeans(n_clusters=2, **common_params).fit_predict(X)
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred)
axs[0, 0].set_title("Non-optimal Number of Clusters")

# Predictions von X_aniso erstellen mit KMeans-, n_clusters=3
y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X_aniso)
axs[0, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
axs[0, 1].set_title("Anisotropically Distributed Blobs")

# Predictions von X_varied erstellen mit KMeans-, n_clusters=3
y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X_varied)
axs[1, 0].scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
axs[1, 0].set_title("Unequal Variance")

# Predictions von X_filtered erstellen mit KMeans-, n_clusters=3
y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X_filtered)
axs[1, 1].scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
axs[1, 1].set_title("Unevenly Sized Blobs")

plt.suptitle("Unexpected KMeans clusters").set_y(0.95)
plt.show()

# %%        
###     Possible solutions      ###
# For an example on how to find a correct number of blobs, see: silhouette analysis
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py    . In this case 

# it suffices to set n_clusters=3.

# Predictions von X durch N-clusters=3 optimieren
y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Optimal Number of Clusters")
plt.show()

# %%        Zufallsinitialisierungen erhöhen, n_init

# Um mit ungleichmäßig großen Blobs umzugehen, kann man die Anzahl der
# Zufallsinitialisierungen erhöhen. In diesem Fall setzen wir n_init=10, 

# Predictions von X_filtered durch n_init=10 optimieren
y_pred = KMeans(n_clusters=3, n_init=10, random_state=random_state).fit_predict(
    X_filtered
)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs \nwith several initializations")
plt.show()

# %%        GaussianMixture 
# ... das ebenfalls von gaußförmigen Clustern ausgeht, aber keine Beschränkungen
# für ihre Varianzen auferlegt.
# "Anisotropically Distributed Blobs" und "Unequal Variance"

from sklearn.mixture import GaussianMixture

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Predictions für X_aniso durch GaussianMixture optimieren
y_pred = GaussianMixture(n_components=3).fit_predict(X_aniso)
ax1.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
ax1.set_title("Anisotropically Distributed Blobs")

y_pred = GaussianMixture(n_components=3).fit_predict(X_varied)
ax2.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
ax2.set_title("Unequal Variance")

plt.suptitle("Gaussian mixture clusters").set_y(0.95)
plt.show()

# %%
###     Comparison of the K-Means and MiniBatchKMeans clustering algorithms     ###
# ----------------------------------------------------------------------------- 

# compare the performance of the MiniBatchKMeans and KMeans: 
# the MiniBatchKMeans is faster, but gives slightly different results

# generating the blobs of data to be clustered.

import numpy as np

from sklearn.datasets import make_blobs

np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=30000, centers=centers, cluster_std=0.7)

# %% Vergleichen KMeans vs. MiniBatchKMeans

# Compute clustering with KMeans

import time

from sklearn.cluster import KMeans

k_means = KMeans(init="k-means++", n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

# Compute clustering with MiniBatchKMeans

from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(
    init="k-means++",
    n_clusters=3,
    batch_size=batch_size,
    n_init=10,
    max_no_improvement=10,
    verbose=0,
)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0


# %%        Parität(Gleichsetzung, -stellung, [zahlenmäßige] Gleichheit) zwischen den Clustern herstellen

from sklearn.metrics.pairwise import pairwise_distances_argmin

k_means_cluster_centers = k_means.cluster_centers_
order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
mbk_means_cluster_centers = mbk.cluster_centers_[order]

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

# %%        Plotting the results

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_batch, k_means.inertia_))

# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("MiniBatchKMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %f" % (t_mini_batch, mbk.inertia_))

# Initialize the different array to all False
different = mbk_means_labels == 4
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += (k_means_labels == k) != (mbk_means_labels == k)

identical = np.logical_not(different)
ax.plot(X[identical, 0], X[identical, 1], "w", markerfacecolor="#bbbbbb", marker=".")
ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
ax.set_title("Difference")
ax.set_xticks(())
ax.set_yticks(())

plt.show()


# %%        DBSCAN-Algorithmus

"""
Der DBSCAN-Algorithmus betrachtet Cluster als Bereiche mit hoher Dichte, die durch
 Bereiche mit geringer Dichte getrennt sind. Aufgrund dieser eher allgemeinen 
 Sichtweise können die von DBSCAN gefundenen Cluster eine beliebige Form haben, 
 im Gegensatz zu k-means, das davon ausgeht, dass die Cluster konvex geformt sind. 
 Die zentrale Komponente von DBSCAN ist das Konzept der Kernproben, d. h. Proben, 
 die sich in Bereichen mit hoher Dichte befinden. Ein Cluster besteht daher aus 
 einer Reihe von Kernproben, die jeweils nahe beieinander liegen (gemessen durch 
 ein Abstandsmaß), und einer Reihe von Nicht-Kernproben, die nahe an einer Kernprobe 
 liegen (aber selbst keine Kernproben sind). Es gibt zwei Parameter für den Algorithmus, 
 
 min_samples und eps, 
 
 (min_samples: minimum number of points  
 eps: distance to points)
 
 die formell definieren, was wir meinen, wenn wir von Dichte sprechen. 
 Höhere min_samples oder niedrigere eps stehen für eine höhere Dichte, die zur 
 Bildung eines Clusters erforderlich ist.
 
 * Während der Parameter min_samples in erster Linie steuert, wie tolerant der 
 Algorithmus gegenüber Rauschen ist (bei verrauschten und großen Datensätzen 
 kann es wünschenswert sein, diesen Parameter zu erhöhen), ist der Parameter eps 
 von entscheidender Bedeutung, um ihn für den Datensatz und die Abstandsfunktion 
 angemessen zu wählen, und kann normalerweise nicht auf dem Standardwert belassen werden. 
 
 """
 
im = Image.open("sphx_glr_plot_dbscan_002.png")
display(im)

# %%    use make_blobs to create 3 synthetic clusters.

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

# %%    visualize the resulting data:

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# %%    mit labels_ Zugriff 
# Auf die von DBSCAN zugewiesenen Labels kann man mit dem Attribut labels_ zugreifen.
# Verrauschten Proben wird das Label math:-1 zugewiesen.

import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# %%    Bewertungsmetriken  in DBSCAM
# Beispiele für solche Metriken sind Homogenität, Vollständigkeit, V-Maß,
# Rand-Index, angepasster Rand-Index und angepasste gegenseitige Information (AMI).

print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
print(
    "Adjusted Mutual Information:"
    f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
)
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

# %%    Plot results
# Kernproben (große Punkte) und Nicht-Kernproben (kleine Punkte) sind entsprechend
# dem zugewiesenen Cluster farblich kodiert. Proben, die als Rauschen gekennzeichnet
# sind, werden in Schwarz dargestellt.

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()