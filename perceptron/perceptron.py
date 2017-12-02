import numpy as np

class Perceptron:
	"""
	Base class for differnet types of Perceptron
	"""
	def __init__(self, l_rate=0.01, n_iter=10):
		"""
		Initilize learning rate and desired number of iterations
		"""
		self.learning_rate = l_rate
		self.n_iterations = n_iter

	def net_input(self, X):
		"""
		Calculate net input
		"""
		return np.dot(X, self.weights[1:]) + self.weights[0]

	def activation(self, X):
		"""
		Activation function for input vector X
		"""
		return self.net_input(X)
	
	def predict(self, X):
		"""
		Predict outcome for input vector X
		"""
		return np.where(self.activation(X) >= 0.0, 1, -1)


class SimplePerceptron(Perceptron):

	def fit(self, X, Y):
		"""
		fit the samples X using simple perceptron approach
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

class AdalinePreceptron(Perceptron):
	
	def fit(self, X, Y):
		"""
		Fit samples X using Adaline perceptron aproach
		"""
		self.weights = np.zeros(1 + X.shape[1])
		self.cost = list()

		for i in range(0, self.n_iterations):
			output = self.predict(X)
			errors = (Y - output)
			self.weights[1:] = self.weights[1:] + (self.learning_rate * X.T.dot(errors))
			self.weights[0] = self.weights[0] + (self.learning_rate * errors.sum())
			c = (errors**2).sum() / 2.0
			self.cost.append(c)
		
		return self

