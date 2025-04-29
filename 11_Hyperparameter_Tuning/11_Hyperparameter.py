# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:34:35 2025

@author: swett
"""
#%%

from bayes_opt import BayesianOptimization
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
import numpy as np
import os
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV,\
    RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold,\
    train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import time
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

#%%

# Random Search

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = pd.read_csv(url, header=None)

print(dataframe)

# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

print(X)
print(y)

# define model
model = LogisticRegression()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)

print(space)

# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy',
                            n_jobs=-1, cv=cv, random_state=1, verbose=1)

# execute search
result = search.fit(X, y)

# summarize result
print(f'Best Score: {result.best_score_}')
print(f'Best Hyperparameters: {result.best_params_}')

#%%

# Grid Search

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = pd.read_csv(url, header=None)

# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# define model
model = LogisticRegression()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

# define search
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X, y)

# summarize result
print(f'Best Score: {result.best_score_}')
print(f'Best Hyperparameters: {result.best_params_}')



#%%

# Random Search for Regression

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = pd.read_csv(url, header=None)

# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# define model
model = Ridge()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]
space['positive'] = [True, False]

# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv, random_state=1)

# execute search
result = search.fit(X, y)

# summarize result
print(f'Best Score: {result.best_score_}')
print(f'Best Hyperparameters: {result.best_params_}')


#%%

# Grid Search for Regression

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = pd.read_csv(url, header=None)

# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# define model
model = Ridge()

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['positive'] = [True, False]

# define search
search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)

# execute search
result = search.fit(X, y)

# summarize result
print(f'Best Score: {result.best_score_}')
print(f'Best Hyperparameters: {result.best_params_}')


#%% Hyperopt

# Hyperopt ist eine Python-Bibliothek zur automatisierten Hyperparameter-Optimierung. 
# Sie hilft dabei, die besten Parameter für Machine-Learning-Modelle oder andere 
# Optimierungsprobleme zu finden – effizienter und oft besser als manuelles Ausprobieren 
# oder Gitter-/Random-Suche.

# Hyperopt basiert auf Bayesian Optimization und nutzt 
# Tree-structured Parzen Estimator (TPE) oder Random Search, 
# um den Suchraum intelligent zu durchsuchen.


# Preprocessing
# load data
d1 = pd.read_csv("11_Hyperparameter_Tuning/mobile_price_data_train.csv")
d2 = pd.read_csv("11_Hyperparameter_Tuning/mobile_price_data_test.csv")

data = pd.read_csv("11_Hyperparameter_Tuning/mobile_price_data_train.csv")

# read data
print(data.head())

# show shape
print(data.shape)

# show list of columns
print(list(data.columns))

# split data into features and target
y = data.price_range.values
X = data.drop("price_range", axis=1).values


# standardize the feature variables 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400, 500, 600]),
    "max_depth": hp.choice('max_depth', np.arange(1, 16, dtype=int)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

# define objective function
def hyperparameter_tuning(params):
    clf = RandomForestClassifier(**params,n_jobs=-1)
    acc = cross_val_score(clf, X_scaled, y, scoring="accuracy", n_jobs=-1).mean()
    return {"loss": -acc, "status": STATUS_OK}

# Initialize trials object
trials = Trials()

best = fmin(
    fn=hyperparameter_tuning,
    space=space, 
    algo=tpe.suggest, 
    max_evals=50, 
    trials=trials
)

print(f"Best: {format(best)}")


#%%
# Bayesian Optimization

# TODO

# # Preprocessing
# # Define score
# acc_score = make_scorer(accuracy_score)

# # Load dataset
# trainSet = pd.read_csv("./datasets/tabular-playground-series-apr-2021/train.csv")
# testSet = pd.read_csv("./datasets/tabular-playground-series-apr-2021/test.csv")

# print(trainSet.head())

# # Remove not used variables
# train = trainSet.drop(columns=['Name', 'Ticket'])
# train['Cabin_letter'] = train['Cabin'].str[0:1]
# train['Cabin_no'] = train['Cabin'].str[1:]

# print(train.head())

# # Feature generation: training data
# train = trainSet.drop(columns=['Name', 'Ticket', 'Cabin'])
# train = train.dropna(axis=0)
# train = pd.get_dummies(train)

# print(train.head())

# # train validation split
# X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=['PassengerId','Survived'], axis=0),
#                                                   train['Survived'],
#                                                   test_size=0.2, random_state=111,
#                                                   stratify=train['Survived'])


#%%

# # Hyperparameter Tuning


# # Define Function
# # Gradient Boosting Machine
# def gbm_cl_bo(max_depth, max_features, learning_rate, n_estimators, subsample):
#     params_gbm = {}
    
#     params_gbm['max_depth'] = round(max_depth)
#     params_gbm['max_features'] = max_features
#     params_gbm['learning_rate'] = learning_rate
#     params_gbm['n_estimators'] = round(n_estimators)
#     params_gbm['subsample'] = subsample
    
#     scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gbm), X_train, y_train, scoring=acc_score, cv=5, n_jobs=-1).mean()))
    
#     return scores