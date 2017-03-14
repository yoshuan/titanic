# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:52:54 2017

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import math
from collections import Counter
import os
from sklearn import cross_validation
import matplotlib.pyplot as plt

###
#Import Data
###
os.chdir('D:\\learnPython\\Titanic')
data = pd.read_csv('train.csv')
#X = data.drop('Survived', axis = 1)
#y = data['Survived']

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
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Title'] = le.fit_transform(data['Title'])
data['Deck'] = le.fit_transform(data['Deck'])

#Discard Unused Columns
data = data.drop(['Name','Ticket','Cabin','PassengerId'], axis = 1)
#Train Test Split
train, intermediate_set = cross_validation.train_test_split(data, train_size=0.6, test_size=0.4)
cv, test = cross_validation.train_test_split(intermediate_set, train_size=0.5, test_size=0.5)


###
#Data Modeling
###










