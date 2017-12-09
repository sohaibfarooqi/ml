import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import numpy as np

from .util import plot_decision_regions

iris = datasets.load_iris()
x = iris.data[:,[2,3]]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

ppn = Perceptron(max_iter=10, eta0=0.1, random_state=0)
ppn.fit(x_train_std, y_train)

y_predict = ppn.predict(x_test_std)
print("Misclassified Samples: {}".format((y_test != y_predict).sum()))
print("Accuracy Score: {}".format(accuracy_score(y_test, y_predict)))

x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(x_combined_std, y_combined, ppn)

plt.show()

