import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from .sbs import SBS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(x_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()