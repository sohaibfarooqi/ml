import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
	"""
	Utility function to plot decision boundry
	"""
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

class Preceptron:
	"""
	This class implements unit step preceptron
	"""
	def __init__(self, learning_rate=0.01, n_iterations=10):
		"""
		Initilize learning rate and number of iterations
		"""
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations

	def fit(self, X, Y):
		"""
		fit the samples X
		"""
		self.weights = np.zeros(1 + X.shape[1])
		self.errors = list()

		for i in range(0, self.n_iterations):
			err = 0
			for xi, target in zip(X, Y):
				update = self.learning_rate * (target - self.predict(xi))
				self.weights[1:] = self.weights[1:] + (update * xi)
				self.weights[0] = self.weights[0] + update
				err = err + int(update != 0.0)
			self.errors.append(err)
		return self

	def net_input(self, X):
		"""
		Net input equals dot product of X and weights vector plus threshold
		Note. adding scalar and vector in numpy behaves differently
		
		a = [1, 1, 1 ,1, 1]
		ar = numpy.array(a)
		print ar + 2
		[3, 3, 3, 3, 3]
		"""
		return np.dot(X, self.weights[1:]) + self.weights[0]

	def predict(self, X):
		"""
		Predict the outcome
		"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)



df =  pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

plt.figure(100)
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')

plt.scatter(X[50:100, 0], X[50:100, 1],
             color='blue', marker='x', label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc="upper left")


ppn = Preceptron(learning_rate=10, n_iterations=100)
ppn.fit(X, y)

plt.figure(200)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')

plt.figure(300)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

