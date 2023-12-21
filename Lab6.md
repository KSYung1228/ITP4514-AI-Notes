# Laab6 Classification
## Type of Machine Learning
 - Supervised learning - trained with input and output data
 - Unsupervised learning - trained with input data only
 - Semi-supervised learning - trained with large amount of un-labelled data + smal amont of labelled data
 - Reinforcement learning - traning reinforced by positive and negative rewards in the form of weights or scores

## Regression vs Classification
 - Regression
   - Predicted output is a continuous value
   - Examples of output values:
     - income
     - temperature
     - height
     - volume
 - Classification
 - Predict output is a categorical variable in the form of class labels. It can be brinary or multiclass
 - Examples of output class labels:
   - Yes/No
   - Grade A/B/C/D/E
   - dayOfWeek Mon/Tue/wed...
   - Gender

## Classification Algorithms
 - Logistic regression
 - K-Nearest Neighbours(KNN)
 - Naive Bayes
 - Decision Tree
 - Support Vector Machine(SVM)
 - Neural Networks(NN)

## Common Classification Algorithms
 - There are many important classification algorithms
 - No single method will do well in all classification problems
 - Methods based on several classification algorithms:
   - Voting classification
   - Ensemble methods

## Logistic Regression
 - uses a logistic function to model a binary dependent variable
 - Standard logistic function
 - $\delta:\mathbb{R}\rightarrow(0,1)$ is defined as follows:
 - $\delta(t) = \frac{e^t}{e^t+1} = \frac{1}{1+e^{-1}}$
 - Logistic regression is a classification algorithm used to find the probability of event success and event failure. it is used when the dependent variable is binart in nature (e.g., 0/1, true/false, yes/no, pass/fail)
 - Logistic Regression Curve
![](/Lab6/Picture1.png)
 - Application example:
   - Predict disease outcome based on observed symptoms of patients
   - Predict whethwe there will be system failure based on machine running statistics
   - Predict whether a mortgage will become default(cannot be repaid) based on economic and borrower's data
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# make 3-class dataset for classification
centers = [[-5,0],[0, 1.5],[5, -1]]
X,y = make_blobs(n_samples = 1000, centers = centers, random_state = 40)
transformation = [[0.4,0.2],[-0.4,-1.2]]
X = np.dot(X, transformation)

clf = LogisticRegression(solver = 'sag', max_iter=100, random_state=42).fit(X,y)

# create a mesh to plot in
h = 0.2 #step size to mesh
x_min,x_max = X[:,].min() -1, X[:,0].max()+1
y_min, y_max = y[:,].min() -1, X[:,0].max() +1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Plot the decision boundary. for that, we will assign a color to each point in the mesh[x_min,x_max]x[y_min,y_max]

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx,yy,Z,cmap = plt.cm.Paired)
plt.title("Decision surface of LogisticRegression(%s)")
plt.axis('tight')

# plot also the traning points
colors = "bry"
for i, color in zip(clf.classes_, colors):
  idx = np.where(y==i)
  plt.scatter(X[idx, 0], X[idx, 1], c = color, cmap = plt.cm.Paired, edgecolor = 'black', s = 20)

# Plot the three one-against-all classifiers
xmin,xmax = plt.xlim()
ymin,ymax = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_

def plot_hyperplane(c, color):
  def line(x0):
    return (-(x0*coef[c,0])-intercept[c])/coef[c,1]
  plt.plot([xmin,xmax],[line(xmin),line(xmax)],ls = "--", color = color)

for i, color in zip(clf.classes_, colors):
  plot_hyperplane(i,color)
plt.show()
```
![](/Lab6/Picture2.png)

## Advantages and Disadvantages of Logistic Regression
 - Advantages
   - simple to implement
   - Fast training
 - Disadvantages
   - Overfitting(If number of observations is less than the number of input features)


## K-Nearest Neighbours(KNN)
 - Assign the data to one of the classes(caregories) based on the average values of its K nearest neighbours
 - Weights can be assigned to the contribution of the neighbours
 - KNN is sensitive to the local structure of the data

```py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

n_neighbors = 15
for weights in ['uniform', 'distance']:
  clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights)
  clf.fit(X,y)

h = .02 #strp size in the mesh
# Create color maps
cmap_light = ListedColormap(['orange','cyan','cornflowerblue'])
cmap_bold = ['darkorange','c','darkblue']

# Plot the decision boundary. for that, we will assign a color to each point in the mesh[x_min,x_max]x[y_min,y_max]
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,0].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap = cmap_light)

# Plot also the traning points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue = iris.target_names[y], palette=cmap_bold, alpha = 1.0, edgecolor = "black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k=%i, weights = '%s')" % (n_neighbors, weights))
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
```
![](/Lab6/Picture3.png)

## Advantages and Disadvantages of KNN
 - Advantages 
   - Does not requires traning period, therefore runs much faster
   - New data can be added without affecting the accuracy of the algorithm
   - Easy to implement
 - Disadvantages
   - Does not work well with largr datasets
   - Does not work well with high dimentions
   - Sensiice to noisy data, missing values and outliers

## Naive Bayes
 - Calssifies according to "Maximum likelihood Principle"
 - calculates likelihood by the Bayes' theorem
 - To classify K possible outcomes, $C_k$  based on some n independent variables $X = (x_1,...,x_n)$, the probability can be wirtten as $p(C_k|x_1,...,x_n)$
 - If X~ multinomial distribution, the posterior probability can be given by :
$$
p(x|C_k) = \frac{(\sum_ix_i)!}{\prod_ix_i!}\prod\limits_{i}P\substack{ki}^{xi}
$$
 - Naive Bayes assumes that all predictors are independent
   - That is, the presence of one particular feature in a class does not affect the presence of another
 - Example:
```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

clf = GaussianNB()
y_pred = clf.fit(X_train, y_train).predict(X_test)
y_pred

print("Number of misabeled points out of a total %d points: %d" %(X_test.shape[0],(y_test != y_pred).sum()))
```
## Advantages and Disadvantages of naive bayes
 - Advantages
   - Simple bot powerful algorithms
   - Suitable for multi-class prediction
   - Suitable for categorical input variables
 - Disadvantages
   - Need to assumes that all predictors are independent
   - Zero-frequency problem for those categories with low frequencies
   - Estimators can be wrong in some cases

## Decision Tree
 - a supervised machine learning algorithm that is used for classification and regression analysis
 - recursively partitioning the input data into smaller subsets based on the values of the input features
   - unti a stopping criterion is met
 - Each internal node of the decision tree represents a test on an input feature
 - Rach leaf node represents a class label or a numerical value
![](/Lab6/Picture4.png)

## Splitting the Decision Tree
 - At each node, the "Best" split(i.e. branch) is obtained by calculating how much "accuracy"(for prediction) each split will gain, the split with maximum gain is chosen
 - Decision trees target variables can be categorical or numerical
 - Examples:
   - Categorical terget variables:
     - An email spam or not spam
   - Numerical target variables:
     - Budget > 10000
     - Age = 25
 - Common metrics foe measuring the "Best" split:
   - IDE3
   - Gini Index
   - Chi-Square
   - Reduction in Variance
## ID3 Algorithm
 - invented by Ross Quinlan, a computer scientist and professor at the University of Sydney, in 1986
 - employs a top-down, greedy search through the space of possible branches with no backtracking
 - uses Entropy and Information Gain to construct a decision tree
 - Entropy:
   - A decisio tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar values(homogenous)
   - ID3 algorithm uses entropy to calculate the homogeneity of a sample
   - ![](/Lab6/Picture5.png)
   - The entropy of a Dataset D:
      $$H(D) = -\sum\limits_{i-1}\limits^n\limits{p_i\log_2p_i}$$
   - There are n classes of samples
   - $p_i$ is the probability of samples belong to cass i
   - H(D) denotes the uncertainty of classifying the dataset D
   - ![](/Lab6/Picture6.png)
- Example:
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()

for pairdx, pair in enumerate([[0,1],[0,2],[0,3],
                               [1,2],[1,3],[2,3]]):
    X = iris.data[:, pair]
    y = iris.target

clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plt.figure()
plot_tree(clf, filled=True)
plt.show()
```
![](/Lab6/Picture7.png)
 - Advantages
   - Easy to understand
   - Easy to generate rules
   - Used for noth classification and regression
   - Non-parametric(i.e. there is no assumption on the distribution)
 - Disadvantages
   - Overfitting
   - Classifies by rectangular partitioning(for which no efficient solution algorithm has been found)
   - Pruning(i.e. selective removal of certain parts) is necessary for large trees

## Support Vector Machine(SVM)
 - Developed at AT & T bell Laboratories by Vladimir Vapnik with colleagues
 - SVMs are one of the most robust prediction methods
 - In classification, the objective of SVM is to find a hyperplane that maximally separates the classes in the feature space
 - SVMs work by finding the best decision boundary that separates the instances belonging to different classes with the maximum margin

## Advantages and Disadvantages of SVM
 - Advantages
   - Can handle data points not linearly sparable
   - Effective in high dimension
   - Robust
 - Disadvantages of SVM
   - Not suitable when nmber of input features
     - numbers of training smaples
   - Overfitting

## Confusion Matrix
 - a table that is used to evaluate the performance of a classification model
   - by comparing the predicted labels to the actual labels
   - Can be used to coun the number of cases predicted correctly or wrongly

## Some terminologies
 - Ture positive(TP)
 - False positive(FP)
 - True negative(TN)
 - False negative(FN)
 - Calculations:
   - Accuracy:
     - $\frac{(TP+TN)}{TP+TN+FP+FN}$
   - Precosopm:
     - $\frac{TP}{(TP+FP)}$
   - Recall:
     - $\frac{TP}{(TP+FN)}$
   - F1 Score:
     - $2\times\frac{precision \times recall}{(precision + recall)}$
   - True Positive Rate:
     - $-\frac{TP}{TP+FP}$
   - False Positive Rate:
     - $\frac{FP}{(TP+FP)}$
![](/Lab6/Picture8.png)
## Example
 - 500 email with know labels
 - TP: 140 emails as spam
 - FP: 180 emails as spam
 - TN: 80 emails as not spam
 - FN: 100 emails as not spam
---
![](/Lab6/Picture9.png){width=450px;}
 ## Multiclass confusion matrix
  - 2x2 Confusion Matrix is suitable for 2-class classification task
![](/Lab6/Picture9.png){width=450px;}