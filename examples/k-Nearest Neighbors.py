import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# this initiate the class KNeighborsClassifier so we can use the prediction model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston


# Load the forge dataset
X, y = mglearn.datasets.make_forge()
# split data set in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# create the Neighbors Classifier model
clf = KNeighborsClassifier(n_neighbors=3)
# load the model with the data
clf.fit(X_train, y_train)
# print the predictions
print("Test set predictions: {}".format(clf.predict(X_test)))
# evalute the model
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# Model visualization. This allow us to see where the model changes from one prediction to the other
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    #  the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()



