import matplotlib
matplotlib.use('TkAgg')

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#load dataset
diabetes = datasets.load_diabetes()

#using only one feature
x_values = diabetes.data[:, np.newaxis, 2]
y_values = diabetes.target

#Split the data into training and testing samples
x_train, x_test = x_values[:-20], x_values[-20:]
y_train, y_test = y_values[:-20], y_values[-20:]

#Train LinearRegression model using training data
#Make predictions using test data
regression = linear_model.LinearRegression()
regression.fit(x_train, y_train)
prediction = regression.predict(x_test)

#Calculate mean squared error, variance and coefficients
print("COEFFICIENTS : {}".format(regression.coef_))
print("MEAN SQUARED ERROR : {}".format(mean_squared_error(y_test, prediction)))
print("VARIANCE SCORE : {}".format(r2_score(y_test, prediction)))

#Plot the training sample and regression line
plt.scatter(x_train, y_train, color="red")
plt.plot(x_test, prediction, color="black")
plt.show()