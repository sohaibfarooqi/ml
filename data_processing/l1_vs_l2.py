import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df.columns = [
		'Class label', 'Alcohol',
		'Malic acid', 'Ash',
		'Alcalinity of ash', 'Magnesium',
		'Total phenols', 'Flavanoids',
		'Nonflavanoid phenols',
		'Proanthocyanins',
		'Color intensity', 'Hue',
		'OD280/OD315 of diluted wines',
		'Proline'
]

x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_norm = sc.fit_transform(x_train)
x_test_norm = sc.transform(x_test)

#Check if coefficients are sparse. L1 regularization can help
#filter some irrelevent features if weight vector is sparse.
lg = LogisticRegression(penalty='l1', C=0.1)
lg.fit(x_train_norm, y_train)
print('Training accuracy:', lg.score(x_train_norm, y_train))
print('Test accuracy:', lg.score(x_test_norm, y_test))
print("Intercepts", lg.intercept_)
print("Coefficients", lg.coef_)

fig = plt.figure()
ax = plt.subplot(111)

colors = [	'blue', 'green', 'red', 'cyan',
			'magenta', 'yellow', 'black',
			'pink', 'lightgreen', 'lightblue',
			'gray', 'indigo', 'orange']
weights, params = [], []

#Prepare data to plot `C` vs weight coefficient
for c in np.arange(0, 10):
	lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
	lr.fit(x_train_norm, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
	plt.plot(params, weights[: ,column], label=df.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

