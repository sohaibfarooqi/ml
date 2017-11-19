import matplotlib
matplotlib.use('TkAgg')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

def true_func(X):
    return np.cos(1.5 * np.pi * X)

n_samples = 30
degrees = [1,4,15]

x = np.sort(np.random.randn(n_samples))
y = true_func(x) + np.random.randn(n_samples) * 0.1

for i in range(0, len(degrees)):
	polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
	linear_regression = LinearRegression()
	pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])

	pipeline.fit(x[:, np.newaxis], y)
	x_test = np.linspace(0, 1, 100)
    
	plt.plot(x_test, pipeline.predict(x_test[:, np.newaxis]), label="Model")
	plt.plot(x_test, true_func(x_test), label="True function")

	plt.scatter(x, y, edgecolor='b', s=20, label="Samples")

plt.xlabel("x")
plt.ylabel("y")
plt.xlim((0, 1))
plt.ylim((-2, 2))
plt.legend(loc="best")
plt.show()
	

