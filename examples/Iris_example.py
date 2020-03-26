import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# this initiate the class KNeighborsClassifier so we can use the prediction model
from sklearn.neighbors import KNeighborsClassifier

# getting the data and printing some info
iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print("Type of target: {}".format(type(iris_dataset['target'])))

# splitting the iris_dataset in train and test
# X mean the data points while y mean the lables of the data points
X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)


# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8,
                         cmap=mglearn.cm3)
plt.show()

# Initiate the class KNeighborsClassifier with one neighbor for searching
knn = KNeighborsClassifier(n_neighbors=1)
# Add the training data to the model with the function fit
knn.fit(X_train, y_train)
# print(knn)

# Suppose we found an iris with the following measurements: 5, 2.9, 1 and 0.2. apply the knn to the data to get
# iris variation
# First convert de data to a shape so the knn can read. For the conversion we use a numpy array. This is a matrix with
# one row so we use two square brackets
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

# We use the function "prediction" to infer the iris variation from X_new
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format( iris_dataset['target_names'][prediction]))

# For testing our model we use the X_test shape. Since we apply the "predict" function to knn, the knn object saves the
# predictions results
y_prediction = knn.predict(X_test)
# this are the predictions from the X_test
print("Test set predictions:\n {}".format(y_prediction))
# Now we calculate the percentage of accurate predictions
print("Test set score: {:.2f}".format(np.mean(y_prediction == y_test)))
# A similar way to get the percentage of accurate predictions
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))