from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from util import plot_decision_regions

lr = LogisticRegression(C=1000.0, random_state = 0)

iris = datasets.load_iris()
x = iris.data[:,[2,3]]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

lr.fit(x_train_std, y_train)

x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(x_combined_std, y_combined, lr)

plt.show()