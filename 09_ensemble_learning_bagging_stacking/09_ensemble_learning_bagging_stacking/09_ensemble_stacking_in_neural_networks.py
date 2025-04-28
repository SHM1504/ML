# -*- coding: utf-8 -*-
"""
# =============================================================================
# Ensemble Stacking in Neural Networks.ipynb
# =============================================================================

Source: https://colab.research.google.com/github/YashK07/Stacking-Ensembling/blob/main/Ensemble_Stacking_in_Neural_Networks.ipynb
"""

#%% imports

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow.keras.backend as K
from keras.callbacks import EarlyStopping
from numpy import dstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.models import Sequential,load_model

#%% load data

df = pd.read_csv('Churn_Modelling.csv')

#%% inspect data

print(df.head())

#%%
"""This data set contains details of a bank's customers and the target variable
is a binary variable reflecting the fact whether the customer left the bank
(closed his account) or he continues to be a customer."""

print(df.info())

#%%
"""There are 3 features with string values."""

print(df.isnull().sum())

#%%
"""Dataset is free of null values."""

print(df['Geography'].value_counts())

#%%
"""Since the number of countries is less and also there is no rank which can
be associated with the customers of different region. Hence we use one hot
encodding."""

#One hot encodding
geo = pd.get_dummies(df['Geography'],drop_first = True)
gen = pd.get_dummies(df['Gender'],drop_first= True)
df = pd.concat([df,gen,geo],axis=1)

#Drop unnecessary data
df.drop(['Geography','Gender','Surname','RowNumber','CustomerId'],axis=1,inplace = True)

print(df)

#%%
"""We need to do data normalization before we send the data to train a
neural network model."""

X = df.drop('Exited',axis=1)
y = df['Exited']

# normalization

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)

#%%
X_final = pd.DataFrame(columns = ['CreditScore','Age','Tenure','Balance',
                                  'NumOfProducts','HasCrCard','IsActiveMember',
                                  'EstimatedSalary','Male','Germany','Spain'],data = X_scaled)

print(X_final.head())

#%% train-test-split

X_train,X_test,y_train,y_test = train_test_split(X_final,y,test_size = 0.30,random_state = 101)

#%% Modeling

model1 = Sequential()
model1.add(Input(shape=(11,)))
model1.add(Dense(50,activation = 'relu'))
model1.add(Dense(25,activation = 'relu'))
model1.add(Dense(1,activation = 'sigmoid'))

print(model1.summary())

#%%
"""Output shape:
  N-D tensor with shape: (batch_size, ..., units).
For instance, for a 2D input with shape (batch_size, input_dim),
the output would have shape (batch_size, units).
"""

print(y.value_counts())

#%% chose f_score

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    print(y_true,y_pred)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

callback = EarlyStopping(monitor="f1_m", patience=5, mode="max")

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history1 = model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,callbacks=callback)

#%%

fig, ax = plt.subplots(dpi=300)
ax.plot(history1.history['loss'])
ax.plot(history1.history['f1_m'])
ax.plot(history1.history['val_loss'])
ax.plot(history1.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()

# save the model
# model1.save('model1.keras')

#%% Train 3 more different models

model2 = Sequential()
model1.add(Input(shape=(11,)))
model2.add(Dense(25,activation = 'relu'))
model2.add(Dense(25,activation = 'relu'))
model2.add(Dense(10,activation = 'relu'))
model2.add(Dense(1,activation = 'sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history2 = model2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,callbacks=callback)

#%%

fig, ax = plt.subplots(dpi=300)
ax.plot(history2.history['loss'])
ax.plot(history2.history['f1_m'])
ax.plot(history2.history['val_loss'])
ax.plot(history2.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()

# model2.save('model2.keras')

#%%

model3 = Sequential()
model1.add(Input(shape=(11,)))
model3.add(Dense(50,activation = 'relu'))
model3.add(Dense(25,activation = 'relu'))
model3.add(Dense(25,activation = 'relu'))
model3.add(Dropout(0.1))
model3.add(Dense(10,activation = 'relu'))
model3.add(Dense(1,activation = 'sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history3 = model3.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,callbacks=callback)

fig, ax = plt.subplots(dpi=300)
ax.plot(history3.history['loss'])
ax.plot(history3.history['f1_m'])
ax.plot(history3.history['val_loss'])
ax.plot(history3.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()

# model3.save('model3.keras')

#%%

model4 = Sequential()
model1.add(Input(shape=(11,)))
model4.add(Dense(50,activation = 'relu'))
model4.add(Dense(25,activation = 'relu'))
model4.add(Dropout(0.1))
model4.add(Dense(10,activation = 'relu'))
model4.add(Dense(1,activation = 'sigmoid'))

model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])
history4 = model4.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,callbacks=callback)

fig, ax = plt.subplots(dpi=300)
ax.plot(history4.history['loss'])
ax.plot(history4.history['f1_m'])
ax.plot(history4.history['val_loss'])
ax.plot(history4.history['val_f1_m'])
plt.legend(['loss','f1_m',"val_loss",'val_f1_m'])
plt.show()

# model4.save('model4.keras')

#%% load the model

dependencies = {
    'f1_m': f1_m
}

# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'model' + str(i + 1) + '.keras'
		# load model from file
		model = load_model(filename,custom_objects=dependencies)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

n_members = 4
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

#%% create stacked model input dataset as outputs from the ensemble

def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat #
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

#%% fit a model based on the outputs from the ensemble members

def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression() #meta learner
	model.fit(stackedX, inputy)
	return model

model = fit_stacked_model(members, X_test,y_test)

#%% make a prediction with the stacked model

def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

# evaluate model on test set
yhat = stacked_prediction(members, model, X_test)
score = f1_m(y_test/1.0, yhat/1.0)
print('Stacked F Score:', score)

#%%
scores = []

for i, m in enumerate(members):
    pred = m.predict(X_test)
    s = f1_score(y_test,pred.round())
    scores.append(s)
    print(f'F-Score of model {i+1} is {s}')

print(f"Stacked Model gives f1 score of {score.numpy().round(4)} which is higher than any other average model ({(sum(scores)/len(scores)).round(4)}).")
