#!/usr/bin/env python
# coding: utf-8

from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


df = pd.read_csv('Iris.csv')
df.head()


df.columns


df.head()


x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]


y = df['Species']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


clf = LinearSVC().fit(x_train, y_train)


y_pred = clf.predict(x_test)


accuracy_score(y_test, y_pred)


y_pred = pd.DataFrame(y_pred)


y_pred.shape


clf.coef_


clf.intercept_


df['Species'].unique()


df['ID'] = df.index
df['Коэффициент'] = round(df['SepalLengthCm'] / df['SepalWidthCm'], 2)
df.head(3)


sns.lmplot(x='ID', y='Коэффициент', data=df, hue='Species', fit_reg=False, legend=False)
plt.legend()
plt.show()





