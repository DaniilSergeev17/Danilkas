#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


df = pd.read_csv('titanic.csv')


df.head()


df.columns


df.shape


df = df.drop(columns=['Passengerid', 'zero', 'zero.1', 'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7', 'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13', 'zero.14', 'zero.15', 'zero.16', 'zero.17', 'zero.18'])


df.head()


cols = df.select_dtypes(exclude=['float']).columns
df[cols]= df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')


df.info()


df.isna().sum()


df = df.fillna(0)


df.describe()


x_train, x_test, y_train, y_test = train_test_split(df[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']], df['2urvived'], test_size=0.2)


classifier = GradientBoostingClassifier()


classifier.fit(x_train, y_train)


classifier.train_score_


y_pred = classifier.predict(x_test)


pred_df = pd.DataFrame({'y_pred':y_pred,
                        'y_test':y_test})


pred_df.sort_index()


accuracy_score(y_pred, y_test)


parameters = {'learning_rate': [0.01,0.02,0.03],
              'subsample'    : [0.9, 0.5, 0.2],
              'n_estimators' : [100,500,1000],
              'max_depth'    : [4,6,8, 10]
                 }
gcv = GridSearchCV(GradientBoostingClassifier(),parameters, cv=5)
gcv.fit(x_train, y_train)


gcv.best_params_


df1 = pred_df[pred_df['y_pred'] == 1.0]
df1.head()
df1.shape


df2 = pred_df[pred_df['y_test'] == 1.0]
df2.head()
df2.shape






