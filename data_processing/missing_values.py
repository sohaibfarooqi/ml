from sklearn.preprocessing import Imputer
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
		1.0,2.0,3.0,4.0
		5.0,6.0,,8.0
		0.0,11.0,12.0,'''

#Read CSV as String, Missing values will be replaced by NaN
df = pd.read_csv(StringIO(csv_data))
#Check how many NaN values in each column
df.isnull().sum()

"""
This Section demonstrate different strategies in dropping `NaN` values
"""
#Drop Rows that contain atleast 1 NaN value
df.dropna()
#Drop column where atleast 1 value in NaN
df.dropna(axis=1)
#Only drop rows where all columns are NaN
df.dropna(how='all')
#Drop rows where 4 or more columns are NaN
df.dropna(thresh=4)
#Drop row if value in column `C` is NaN
df.dropna(subset=['C'])

"""
Imputer module helps in filling missing data
using either mean, median, most_frequest
"""
imr = Imputer(missing_values="NaN", strategy='mean', axis=0)
imr = imr.fit(df)
imr_data = imr.transform(df.values)