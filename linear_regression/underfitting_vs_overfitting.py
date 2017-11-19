"""
This file demostrate underfitting adn overfitting problem.
Example from sklearn: http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
"""
import matplotlib
matplotlib.use('TkAgg')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

def true_func(X):
	"""
	Cos function applied on each sample in `X`
	"""
	return np.cos(1.5 * np.pi * X)

np.random.seed(0)
#Number of samples
n_samples = 30
#Degrees of polynomial
degrees = [1,4,15]

#Input data is sorted array of randomly generated 30 values
x = np.sort(np.random.randn(n_samples))
#Output data is cos(1.5pi * input) + random * 0.1
y = true_func(x) + np.random.randn(n_samples) * 0.1

for i in range(0, len(degrees)):
	#Polynomial features
	polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
	linear_regression = LinearRegression()
	#Pipeline to apply linear and polynomial features simultanously.
	pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])

	pipeline.fit(x[:, np.newaxis], y)
	#Test sample is linear space vector
	x_test = np.linspace(0, 1, 100)
	#Calculate cross validation score
	scores = cross_val_score(pipeline, x[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
	plt.plot(x_test, pipeline.predict(x_test[:, np.newaxis]), label="Model")
	plt.plot(x_test, true_func(x_test), label="True function")

	plt.scatter(x, y, edgecolor='b', s=20, label="Samples")

	plt.xlabel("x")
	plt.ylabel("y")
	plt.xlim((0, 1))
	plt.ylim((-2, 2))
	plt.legend(loc="best")
	plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
	

