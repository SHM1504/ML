# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:30:19 2025

@author: swett
"""

#%%

# Crossvalidation

# Import Libraries

import numpy as np
np.random.seed(42)

from typing import List
from typing import Tuple


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#%%

# Data

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
print(f'x_train shape: {x_train.shape}; x_test.shape: {x_test.shape}' )



#%%

def print_cv_results(scores: List[float]) -> Tuple[float, float]:
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f'Accuraies :\n {scores}')
    print(f'Mean Score :\n {mean_score}')
    print(f'Std Score :\n {std_score}')
    
    return mean_score, std_score


    
#%%
    
def plot_cv_results(scores: List[float], mean_score: float) -> None:
    plt.axhline(mean_score, linestyle = "-", color = 'red')
    plt.plot(range(len(scores)), scores, color = 'blue')
    plt.xlim(0,len(scores)-1)
    plt.ylim(0.85, 1)
    plt.legend(['Validation Scores', 'Mean Score'])
    plt.show()
    
#%%
# N Neighbors = 3

clf = KNeighborsClassifier(n_neighbors= 3)

scores = cross_val_score(clf, x_train, y_train, cv=5, n_jobs=-1)

mean, _ = print_cv_results(scores)
plot_cv_results(scores= scores, mean_score = mean)


# Wir sehen hier die unterschiedlichen accuracy-Werte in Abhängigkeit von den Trainings und Testdaten
# Für die Crossvalidation berechnen wir den Mittelwert

#%%

# Wir testen den Klassifier anhand verschiedener Neighbors:

# N Neighbors = 4
    
clf = KNeighborsClassifier(n_neighbors= 4)

scores = cross_val_score(clf, x_train, y_train, cv=5, n_jobs=-1)

mean, _ = print_cv_results(scores)
plot_cv_results(scores= scores, mean_score = mean)

# N Neighbors = 5
    
clf = KNeighborsClassifier(n_neighbors= 5)

scores = cross_val_score(clf, x_train, y_train, cv=5, n_jobs=-1)

mean, _ = print_cv_results(scores)
plot_cv_results(scores= scores, mean_score = mean)


# N Neighbors = 10
    
clf = KNeighborsClassifier(n_neighbors= 10)

scores = cross_val_score(clf, x_train, y_train, cv=5, n_jobs=-1)

mean, _ = print_cv_results(scores)
plot_cv_results(scores= scores, mean_score = mean)

#%%

# Grid Search

from sklearn.model_selection import GridSearchCV

parameters = {
    'n_neighbors': [3, 5, 7, 9],
    'weights' : ['uniform', 'distance']
    }

clf = KNeighborsClassifier()
grid_cv = GridSearchCV(clf, parameters, cv = 10) # cv= 10 wird jedes Modell mit 10 Crossvalidation Durchläufen trainiert
grid_cv.fit(x_train, y_train)



#%%

# Print results

print('GridSearch Keys: ')

for key in grid_cv.cv_results_.keys():
    print(f'\t{key}')
    
    
print('----------------------------------------------------------------------')
print()
print()

print('GridSearch params: ')

for param in grid_cv.cv_results_['params']:
    print(f'\t{param}')

#%%

# Best Results

print(f"Best parameters set found on development set: {grid_cv.best_params_}\n")

means = grid_cv.cv_results_["mean_test_score"]
stds = grid_cv.cv_results_["std_test_score"]

for mean, std, params in zip(means, stds, grid_cv.cv_results_["params"]):
    print(f"{mean:.3f} (+/-{2 * std:.3f}) for {params}")

    
#%%

# Best Model

    
clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Accuracy: {score}")
    
#%%

# Random Search

from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_neighbors": sp_randint(1, 15),
    "weights": ["uniform", "distance"],
}
n_iter_search = 10

clf = KNeighborsClassifier()
rand_cv = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=3,
)
rand_cv.fit(x_train, y_train)

#%%


print("RandomSearch Keys:")
for key in rand_cv.cv_results_:
    print(f"\t{key}")
    
print('----------------------------------------------------------------------')
print()
print()

print('RandomSearch params: ')

for param in grid_cv.cv_results_['params']:
    print(f'\t{param}')
    
    
#%%
# Best Results

print(f"Best parameters set found on development set: {rand_cv.best_params_}\n")

means = rand_cv.cv_results_["mean_test_score"]
stds = rand_cv.cv_results_["std_test_score"]

for mean, std, params in zip(means, stds, rand_cv.cv_results_["params"]):
    print(f"{mean:.3f} (+/-{2 * std:.3f}) for {params}")


#%%

# Best Model

clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Accuracy: {score}")
    
#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    