----------------
Machine learning
----------------
**Supervised**

You know the input and output of the processes. For example:
1. Hand written postal code. You need to read all hand written postal code and assign each to number in a computer. 
You can train the algorithm for recognizing the data

**Unsupervised**

You only know the input but no the output. For example:
1. Topics in a set of blogs. You have all the blog at your disposal, but not know which are the topics, 
you need to figure out the topics

**Describing the data**

Matrix like. Each row is the data and the columns are a description of the data. For example
1. Users are described by age, name, shopping list
2. Images are described by grayscale, size, shape

**Categories**
1. Rows are know as _sample_. This are commonly defined with the variable "X"
2. Columns are _features_. this are commonly defined with the variable "y"
3. The _shape_ of the data is the number of _samples_ multiplied by the number of _features_ 
4. The _labels_ are the categories of our _samples_. For example, flower species, breads of dogs
5. The _target_ is the classification of our date set. The _target_ is the whole set of _labels_
6. _Generalization_ is the term used for saying that our model performs well on the real-world 

The data is divide in two parts
1. Training set. For training our model
2. Test set. For testing our model

**Tricks**
1. Visualize the data to find inconsistencies. Use a scatter plot
2. In general, complex models does not work well but the same happens to "too simple" models
3. When model are too complex and highly tied to the test data we say that our "model is overffited", i.e., 
our model will only work well on the train data but perform poorly on the real-world. The same happens to simple models
and we called this "underfitted" 
4. More data yields to complex and better models. Don't be sacared is your model is too complex but you have tons of 
data
5. In K Neighbors Classifier a low number of neighbors correspond to a high complexity model (the model is too 
correlated with a single data point). On the other hand is you choose too many neighbors the model will be 
oversimplified. Select the exact number of neighbors is essential in theses types of models  

**Types of problems**
1. Classification. We we want to predict outcomes in a non continue space. For example, predict: languages, species,
yes-no answers
2. Regression. We we want to predict outcomes in a continue space. For example, predict: wages, prices, weights



