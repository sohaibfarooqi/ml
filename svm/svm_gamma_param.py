import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from util import plot_decision_regions

np.random.seed(0)

x_xor = np.random.randn(200,2)
y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm.fit(x_xor, y_xor)

plot_decision_regions(x_xor, y_xor, classifier=svm)

svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=10.0)
svm.fit(x_xor, y_xor)

plot_decision_regions(x_xor, y_xor, classifier=svm)
plt.show()