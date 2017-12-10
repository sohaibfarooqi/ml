from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

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

mms = MinMaxScaler()
x_train_norm = mms.fit_transform(x_train)
x_test_norm = mms.transform(x_test)

sc = StandardScaler()
x_train_norm = sc.fit_transform(x_train)
x_test_norm = sc.transform(x_test)
