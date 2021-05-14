#importing librairies

import pandas as pd
import numpy as np

#loading dataset

dataset = pd.read_csv('data.csv')

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#segregating dataset into X and Y

X=dataset.iloc[:,:-1].values
X
Y=dataset.iloc[:,-1].values
Y

#splitting the dataset into test and train

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#training dataset using Support Vector Regression(SVR)

from sklearn.svm import SVR
model1= SVR()
model2= SVR(kernel='linear', degree=2, gamma='scale', coef0=0.001, tol=0.0001, C=1.0, epsilon=0.01, shrinking=True, cache_size=500, verbose=False, max_iter=-1)
model3= SVR(kernel='poly', degree=3, gamma='scale', coef0=0.001, tol=0.0001, C=1.0, epsilon=0.01, shrinking=False, cache_size=500, verbose=True, max_iter=-1)

model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
model3.fit(X_train, Y_train)

#prediction for all test data for validation

ypred1 = model1.predict(X_test)
ypred2 = model2.predict(X_test)
ypred3 = model3.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error

mse1 = mean_squared_error(Y_test,ypred1)
rmse1 = np.sqrt(mse)
print("For Model 1:")
print("Root Mean Square Error:",rmse1)
r2score = r2_score(Y_test,ypred1)
print("R2Score",r2score*100)

mse2 = mean_squared_error(Y_test,ypred2)
rmse2 = np.sqrt(mse2)
print("For Model 2:")
print("Root Mean Square Error:",rmse2)
r2score = r2_score(Y_test,ypred2)
print("R2Score",r2score*100)

mse3 = mean_squared_error(Y_test,ypred3)
rmse3 = np.sqrt(mse3)
print("For Model 3:")
print("Root Mean Square Error:",rmse3)
r2score = r2_score(Y_test,ypred3)
print("R2Score",r2score*100)