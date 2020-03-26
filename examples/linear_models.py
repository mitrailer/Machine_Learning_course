import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
# this initiate the class LinearRegression so we can use the prediction model
from sklearn.linear_model import LinearRegression

# X, y = mglearn.datasets.make_wave(n_samples=60)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# lr = LinearRegression().fit(X_train, y_train)
#
# # The coef is the slope of the line (w)
# print("lr.coef_: {}".format(lr.coef_))
# # The intercept is the offset of the line (b)
# print("lr.intercept_: {}".format(lr.intercept_))
#
# Both scores are low and close, is possible we are underfitting
# print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# # The score of the test set is the R^2 of the linear regression model.
# print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
# Both scores are to far away and the train score is performing too well, is possible we are overfitting
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))