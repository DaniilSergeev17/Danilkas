#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')


df = pd.read_csv('USA_Housing_linear_regression.csv')


df = df.drop(columns=['Address'])
df.head()


df.shape


x_train, x_test, y_train, y_test = train_test_split(df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area House Age', 'Avg. Area Number of Bedrooms', 'Area Population']], df['Price'], test_size=0.2)


regressor = LinearRegression()


regressor.fit(x_train, y_train)


regressor.coef_


y_pred = regressor.predict(x_test)


print(round(mean_squared_error(y_pred, y_test)))


print(round(mean_absolute_error(y_pred, y_test)))


pred_df = pd.DataFrame({'y_pred':y_pred,
                        'y_test':y_test})


pred_df


pred_df = pred_df.sort_index()


pred_df.plot(y='y_pred', kind='line', figsize=(17, 8))
pred_df.plot(y='y_test', kind='line', figsize=(17, 8), color='red')
plt.show()


plt.figure(figsize=(10, 8), dpi=80)
sns.pairplot(pred_df, kind='reg')





