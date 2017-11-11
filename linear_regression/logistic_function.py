"""
This file implements simple linear regression with one variable. Output Y is plotted against one feature X.
MSE and R^2 are calculated to determine how well model fits in.

Example from sklearn site: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""
import matplotlib
matplotlib.use('TkAgg')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def model(x):
	"""
	Logistic Regression hypothesis function
	"""
	return 1 / (1 + np.exp(-x))

#Number of samples for this test
n_samples = 100

#Normally distributes 100 data points
X = np.random.normal(size=n_samples)
#Output y, cast to float where value of X > 0
y = (X > 0).astype(np.float)
#Multiply X by 4
X *= 4
#Add [0.3 times normally distributed] data points to get new X.
X += .3 * np.random.normal(size=n_samples)

#Reshape X to 2D array
X = X[: , np.newaxis]
#Fit X, y in logistic regression model
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

#Ravel flattens 2D array to 1D array
plt.scatter(X.ravel(), y, color='black', zorder=20)
#Build linear model for test
X_test = np.linspace(-5, 10, 300)

#Logistic function (Sigmoid function)
loss = model(X_test * clf.coef_ + clf.intercept_)
plt.plot(X_test, loss.ravel(), color='red', linewidth=3)

#Fit X, y in linear regression model
ols = linear_model.LinearRegression()
ols.fit(X, y)

#Plot linear regression, set legend and axis scaling
plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
plt.show()