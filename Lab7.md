# Lab7 - Regression & Clustering
## Supervised learning
 - Given the "right answer" for each example in the data
 - map(x,y)\

## Regression
 - A type of supervised learning
 - A computer program predict the output for the given input
 - Learning algorithms typically output a function. $f(x) = mx + b$
 - Example:
   - Predict the claim amount of insured person
   - Predict the security price
 - Regression mdels are used to predict a continuous value
 - Linear Regression

## Sample Dataset
 - Reflect the features of attribute values in a sample dataset
 - The dependency between attribute values
 - The relationship of sample mapping through functions

## Regression Models
 - Simple regression model - This is the most baxic regression model in which predictions are formed from a single, univariate feature of the data
 - Multiple regression model - As name implies, in this regression model the predictions are formed from multiple featres of the data
   - the satistical model that analysis the linear relationship between a dependent variable with given set of independent cariables
   - Type of superised learning
 - Simple linear Regression algorithm(Assume that 2 variabes are linearly related)
 - Multiple linear Regression algorithm(predictis a response using two or more features)
 - ![](/Lab7/Picture1.png)

## Linear Regression
 - Linear regression is called simple if only working with ine independent variable(x)
 - Formula:$f(x) = mx+b$\

**Cost Function**
 - We can measure the accuracy of our linear regression algorithm using the mean squared error(MSE) cost function
 - MSE measures the average squared distance between the predicted output and the actual output(Label)
![](/Lab7/Picture2.png)
![](/Lab7/Picture3.png)
## Mean Absolute Error(MAE)
 - A measure of errors between paired observations expressing the same phenomenon
 - Less sensitive to outliers
 - Many small errors = one large error
 - Best 0<sup>th</sup> order baseline
$$
 MAE = \frac{1}{n}\sum\limits^n\limits_{i=1}|x_i-x|
$$

## Meas Square Error(MSE)
 - How close a regression line is to a set of points
 - Square value aims to remove any negatice signs
 - Give more weight to larger differences
$$
MSE = \frac{1}{n}\sum\limits^n\limits_{i=1}(y_i-\tilde{y}_i)^2
$$

## Gradient descent
 - To find the coefficients that minimize our error function we will use **gradient descent**
 - Gradient descent is an optimzation algorithm which interatively takes steps to the local minimum of the cost function, i.e., to reduce the cost function
 - **Learning rate** controls how much we should adjust the weights with respect to the loss gradient
![](/Lab7/Picture4.png)

## Multiple Linear Regression
 - Linear regression is called multivariate if you are working with at least two independent variables
 - Each of the independent cariables also called features gets multiplied with a weight which is learned by our linear regression algorithm
 - Formular:
$$
f(x) = b + w_1x_1+w_2x_2+...+w_nx_n = b+\sum{^n_{i=1}}w_ix_i
$$
 - Logistic regression is a model that uses a logistic function to model a dependent variable
 - Like all regression analysis, the logistic regression is a predictive analysis
 - Logistic regression is used to describe data and to explain the relationship between one dependent variable and one or more independent variables.
![](/Lab7/Picture5.png)

## Applicaitons
 - forecasting or predictive analysis
 - Optimization
 - Error correction
 - Economics
 - Finance
```py
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1,1],[1,2],[2,2],[2,3]])
y = np.dot(X, np.array([2,6]))+3
reg = LinearRegression().fit(X,y)

reg.score(X,y)

reg.coef_

reg.intercept_

reg.predict(np.array([[3,5]]))
```

## Unsupervised Learning
 - Unsupervised learning is a machine learning technique, where you do not need to supervise the model
 - The data is unlabeled: we have the nput features X, but we do not have the labels y
 - Allow the model to work on its own to discover information
 - Allows to perform more complex processing tasks comared to supervise learning
 - Although, unsupervised learning can be more unpredictabe compared with other natual learning methods

## Why unsupervised learning
 - Can finds all kind of unknown patterns
 - help to find features which can be useful for categorization
 - taken place in real time, all the input data to be analyzed and labeled in the presence of learners
 - easier to get unabeled data from a computer than labeled data

## Compare Suvervised and Unsupervised Learning
![](/Lab7/Picture6.png){width=450px;}

## Types of Unsupervised Learning
 - Unsuervised learning problems aregrouped into clustering and association problems
 - Clustering is an important concept when it comes to unsupervised learning
 - It mainly deals with finding a sructure or pattern in a collection of uncategorized data
 - Clustering algorithms will process your data and find natural culuster(groups) if they wxist in the data
 - can modify how many clusters of algorithms should identify
 - allows to adjust the granularity of these groups
 - Association rules allow to establish associations amongst data objects inside large databases
 - unsupervised technique is about discovering interesting relationships between variables in large databases
 - Example:
   - people that by a new home most likely to buy new furniture

## Disadvantages of Unsupervised Learning
 - cannot get precise information regarding adta sorting, and the output as data used in unsupervised learning is labeled and not known
 - Less accracy of the results is because the input data is not known and not labeled by people in advance
 - This means that the machine requires to do this itself
 - The spectral classes do not always correspond to informational classes
 - The user needs to spend time interpreting and label the classes which follow that classification
 - Spectral properties of classes can also change over time so you can't hace the same class information whi;e mocing from one image to another

## Clustering
 - Can be considered the most important unsupervised learning problem
 - deals with finding a structure in a collection of unlabeled data
 - A loose deginition of clustering could be "the process of organizing objects into groups whose menbers are similar in some way"
 - A cluter is therefore a collection of objects which are "similar" between them and are "dissimilar" to the objects belonging to other clusters
![](/Lab7/Picture7.png)

## The goal of clustering
 - the goal of clustering is to determine the internal grouping in a set of unlabeled data
 - Can be shown that there is no absolute "best" criterion which would be independent of the final aim of the clustering
 - To find a particulat clustering solution, we need to define the similarity measures for the clusters

## Proximity Measures
 - For clustering, we need to define a proximity measure for two points
 - Proximity here means how similar/dissimilar the samples are with respect to each other
 - Similarity measure S(xi,xk): large if xi,xk are similar
 - Dissimilarity(or distance) measure D(xi,xk): small if xi,xk are similar
 - Various similarity measures:
   - Vectors: Cosine Distance
$$
Similarity(p,q) = cos\theta = \frac{p*q}{||p||||q||} = \frac{\sum\limits_{i = 1}\limits^{n}p_iq_i}{\sqrt{\sum\limits_{i=1}\limits^{n}{p_i}^2}\sqrt{\sum\limits_{i=1}\limits^{n}{q_i}^2}}
$$
 - Points: Euclidean Distance
$$
d(p,q) = d(q,p) = \sqrt{(q_1-p_2)^2+(q_2-p_2)^2+...+(q_n-p_n)^2}= \sqrt{\sum\limits^{n}\limits_{i=1}(q_i-p_i)^2}
$$
![](/Lab7/Picture8.png)
![](/Lab7/Picture9.png)

## applications of Clustering
 - Marketing
 - Biology
 - Libraries
 - Insurance
 - City Planning
 - Earthquake studies

## types of Clustering
 1. Exclusive(Partitioning)
    - In this clustering method, Data are grouped in such a way that one data can belong to one cluster only
    - Example: K-means
 2. Agglomerative(Single Linkage Clustering)
    - In this clustering technique, every data is a cluster
    - The interative unions between the teo nearest clusters reduce the number of clusters
    - Example: Hierarchical clustering
 3. Probability(Expectation-Maximization)
    - This technique uses probability distribution to create the clusters
    - Example: Following keywords,"Man's shoe","women's shoe","women's glove","man's glove" can be clustered into two categories "shoe" and "glove" or "man" and "women"
    - Example: Gaussian Mixture Model(GMM)