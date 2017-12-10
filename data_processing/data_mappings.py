import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame([
		['green', 'M', 10.1, 'class1'],
		['red', 'L', 13.5, 'class2'],
		['blue', 'XL', 15.3, 'class1']
	])

#Define column labels for dataframe
df.columns = ['color', 'size', 'price', 'classlabel']

#Define size mappings
size_mappings = {
	'M': 1,
	'L': 2,
	'XL': 3
}
df['size'] = df['size'].map(size_mappings)

#Define mappings for class labels
class_labels = dict((label, idx) for idx,label in enumerate(np.unique(df['classlabel'])))
# df['classlabel'] = df['classlabel'].map(class_labels)

#Differnt way of specifying class lable using sklean builtin method
df['classlabel'] = LabelEncoder().fit_transform(df['classlabel'].values)

#One hot encoder tp transform string features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
ohe = OneHotEncoder(categorical_features=[0])
ohe = ohe.fit_transform(X).toarray()

#Similar to one hot encoder
df = pd.get_dummies(df[['price', 'color', 'size']])
