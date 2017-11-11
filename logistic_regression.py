import matplotlib
matplotlib.use('TkAgg')

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

yeast = datasets.fetch_mldata("yeast")
x_values = yeast.data
y_values = yeast.target

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=.2, random_state=0)



