# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:52:54 2017

@author: HP
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from collections import Counter
import os
from sklearn.cross_validation import train_test_split

###
#Import Data
###
os.chdir('D:\\learnPython\\Titanic')
data = pd.read_csv('train.csv')


###
#Data Preprocessing
###

#Extract Title from 'Name'
data['Title'] = data['Name'].str.extract('\s([A-Z][a-z]+\.)\s', expand = True)

def replace_titles(x):
    title = x['Title']
    if title in ['Don.', 'Major.', 'Capt.', 'Jonkheer.', 'Rev.', 'Col.', 'Sir.']:
        return 'Mr.'
    elif title in ['Countess.', 'Mme.']:
        return 'Mrs.'
    elif title in ['Mlle.', 'Ms.', 'Lady.']:
        return 'Miss.'
    elif title =='Dr.':
        if x['Sex'] == 1:
            return 'Mr.'
        else:
            return 'Mrs.'
    else:
        return title
data['Title'] = data.apply(replace_titles, axis=1)

#Replace Nan in 'Age' group by Title
groupMean = pd.DataFrame(data.groupby(data['Title'])['Age'].mean())
data['Age'] = data.groupby('Title').transform(lambda x: x.fillna(x.mean()))['Age']

#Caculate Family Size by 'SibSp' & 'Parch'
data['FamilySize'] = data['SibSp'] + data['Parch']

#Replace Nan in 'Embarked' Considering 'Ticket', 'S'
Portnan = data[data['Embarked'].isnull()]
checkPC = list()
for i in range(len(data)):
    if data['Ticket'][i][0:2] == '11':
        checkPC.append(data['Embarked'][i])
Counter(checkPC)
data['Embarked'].fillna('S', inplace = True)

#Extract Cabin Area
data['Deck'] = data['Cabin'].str.extract('^([A-Z])', expand = True)
data['Deck'].fillna('Unknown', inplace = True)

#Encode
le = LabelEncoder()
columns = ['Sex','Embarked','Title','Deck']
for col in columns:
    data[col] = le.fit_transform(data[col])

enc = OneHotEncoder(sparse = False)
columns = ['Embarked','Title','Deck']
for col in columns:   
    enc.fit(data[[col]])
    temp = enc.fit_transform(data[[col]])
    temp = pd.DataFrame(temp, columns=[col+'_'+str(i) for i in data[col].value_counts().index])
    temp = temp.drop(temp.columns[[0]], axis = 1)
    data = pd.concat([data,temp], axis = 1)
#Discard Unused Columns
data = data.drop(['Name','Ticket','Cabin','PassengerId','Embarked','Title','Deck'], axis = 1)
#Train Test Split
X = data.drop('Survived', axis = 1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

###
#Modeling
###
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, bootstrap = True, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
