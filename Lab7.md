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

## Hierarchal Clustering: An Example
 - Given a dataset of following points:(1,1),(1,2),(2,4),(5,2),(7,1),(7,3), please conduct a hierarchical clustering(bottom-up) and write down your steps. In each step, please specift the number of clusters, points in each cluster, and centroid of each cluster
 - Initially, each data point is a single cluster
 - Initially, all data points are ventroids
 - How to determine the "nearness" of the clusters
   - Measure cluster distances by distances of centroids
 - How to determine the centroid after two clusters merge?
   - Use their mid point as the new centroid
 - Step 1: Initialize all the controids of cluster
<table><td>centroids</td><td>1,1</td><td>1,2</td><td>2,4</td><td>5,2</td><td>7,1</td><td>7,3</td></table>

 - Step 2: Find the two clusters with the shortest distance
 - Distance formula $d: \sqrt{(x2-x1)^2-(y2-y1)^2}$
 - Using the formula, since (1,2) & (1,1) have the shortest distance with 1, the two centroids merge
 - Thus;
<table>
    <tr>
        <td></td>
        <td>(1,1),</br>(1,2)</td>
        <td>(2,4)</td>
        <td>(5,2)</td>
        <td>(7,1)</td>
        <td>(7,3)</td>
    </tr>
    <tr>
        <td>centroids</td>
        <td>(1,3/2)</td>
        <td>(2,4)</td>
        <td>(5,2)</td>
        <td>(7,1)</td>
        <td>(7,3)</td>
    </tr>
</table>

 - Step 3~N: Find the two clusters with the shortest distance   ***until there is one cluster left***

<table>
    <tr>
        <td></td>
        <td>(1,1),</br>(1,2)</td>
        <td>(2,4)</td>
        <td>(5,2)</td>
        <td>(7,1),</br>(7,3)</td>
    </tr>
    <tr>
        <td>centroids</td>
        <td>(1,3/2)</td>
        <td>(2,4)</td>
        <td>(5,2)</td>
        <td>(7,2)</td>
    </tr>
</table>

<table>
    <tr>
        <td></td>
        <td>(1,1),</br>(1,2)</td>
        <td>(2,4)</td>
        <td>(5,2),</br>(7,1),</br>(7,3)</td>
    </tr>
    <tr>
        <td>centroids</td>
        <td>(1,3/2)</td>
        <td>(2,4)</td>
        <td>(19/3,2)</td>
    </tr>
</table>

<table>
    <tr>
        <td></td>
        <td>(1,1),</br>(1,2),</br>(2,4)</td>
        <td>(5,2),</br>(7,1),</br>(7,3)</td>
    </tr>
    <tr>
        <td>centroids</td>
        <td>(4/3,3/2)</td>
        <td>(19/3,2)</td>
    </tr>
</table>


<table>
    <tr>
        <td></td>
        <td>(1,1),</br>(1,2),</br>(2,4),</br>(5,2),</br>(7,1),</br>(7,3)</td>
    </tr>
    <tr>
        <td>centroids</td>
        <td>(23/6,13/6)</td>
    </tr>
</table>

## Clustering Algorithms in Scikit-learn
 - Partitioning - K-means
 - Single Linkage Clustering(Slc)
 - Expectation Maximization(EM)

## Partitioning - K-means
 - K-means ckustering is an interatice algorithm that partitions a group of data containing n values into k subgroups
 - Each f the n value belongings to the k cluster with the nearest mean

***Algorithm for K-means Clustering***
 1. initialize and select the k-points. These k-points are the means
 2. Use the Euclidean distance to find data-points taht are closest to thrie centrois of the cluster
 3. Calculate the mean of all the points in the cluster which is finding their centroid
 4. repeat setp 1, 2, and 3 until all the centroids remain unchanged

```py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

np.random.seed(0)

batch_size = 45
centers = [[1,1],[-1,-1],[1,-1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples = 3000, centers = centers, cluster_std = 0.7)

kmeans = KMeans(n_clusters = 3, random_state = 0).fit(X)

kmeans.labels_

kmeans.predict([[0,0],[-1,-2]])

kmeans.cluster_centers_

fig = plt.figure(figsize = (8,3))
fig.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.05, top = 0.9)
colors = ['red','green','blue']

k_means_cluster_centers_ = kmeans.cluster_centers_

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

ax = fig.add_subplot()

for k, col in zip(range(n_clusters), colors):
  my_members = kmeans.labels_ == k
  cluster_center = kmeans.cluster_centers_[k]
  ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
  ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
```

## Single Linkage Clustering(SLC)
 - has a "hierarchical agglomerative cluster structure" as it "links" each point of the dataset to one another and therefore separates them in different clusters
 - Hierarchical clustering is used to cluster unlabeled data points
 - Like K-means clustering, hierarchical clustering also groups together the data points with similar characteristics
 - In some cases the result of hierarchical and K-means clustering can be similar
 - Two types of hierarchical clustering:Agglomerative and Divisive
 - In the former, data points are clustered using a bottom-up appreach starting with individual data points
 - Divisive is a top-down approach where all the data points are treated as one big cluster and the clustering process involves dividing the one big cluster into several small clusters
 - Here we will focus on agglomerative clustering that involves the bottom-up approach

***Algorithm for Hierarchical Clustering***
 1. Consider each object as a cluster(there will be n clusters for n objects)
 2. form a cluster by joining the two closest data points resulting in n-1 clusters
 3. form more clusters by joining the two closest clusters resulting in n-2 clusters
 4. Repeat the above three steps until one big cluster is formed

![](/Lab7/Picture10.png){width=450px;}

```py
from sklearn.cluster import AgglomerativeClustering
import numpy as np
X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])
clustering = AgglomerativeClustering().fit(X)

clustering.labels_
```

## Expectation Maximization
 - Clustering is an essential part of ant data analysis. using an algotithm such as K-means leads to hard assignments, meaning that each point is definitively assigned a cluster center
 - This leads to some interesting problems: what it the true cluster actually overlap
 - What about data that is mre spread out; how do we assign clusters then
 - Gaussian Mixture Models save the day
 - Traning these models requires using a very famous algorithm called the Expectation Maximization Alogrithm
 - This algorithm is a soft clustering, which uses probability to make that point as "shared" between the two clusters
 - while k-mean is doing a deterministic assignment of the data, EM is instead using a probabilistic approach
 - This algorithm iterates two steps, defined by estimation and maximiaation
    1. The estimation step finds the likelihood that a data "i" comes from a cluster "j"
    2. The maximization step uses this "expected" model and maximizes it to find the data fits
 - Gaussian Distribution is the most famous and important of all statistical distributions
 - Here's an sxample of a Gaussian centered(mean) at 0 with a standard deviation of 1
![](/Lab7/Picture11.png)
 - There is a famous theorem in statics called the Central Limit Theorem that states that enough random sample from any distribution tend to resemble a normal(Gaussian) distribution
 - This makes Gaussian very powerful and versatile
```py
import numpy as np
from sklearn.mixture import GaussianMixture
X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
gm = GaussianMixture(n_components=2, random_state=0).fit(X)

gm.means_

gm.predict([[0,0],[12,3]])
```