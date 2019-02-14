#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### DESCRIPTIVES AND MODELS #####


# In[2]:


### import libraries
import numpy as np
import pandas as pd
from datetime import datetime,date


# In[3]:


start_time = datetime.now()
print('Start time: ', start_time)


# In[4]:


##### LOAD DATA
print('Loading data...')


# In[5]:


df = pd.read_csv('../data/processed_data/session_level_data_final.tsv.gz', compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt'])


# In[6]:


##### DESCRIPTIVES
print('Calculating descriptives...')


# In[7]:


descriptives_dict = {'unique_visitors':df['visitor_id'].nunique(),
                     'visits':df.shape[0],
                     'percentage_purchases':round(df['purchase'].value_counts()[1]/(df['purchase'].value_counts()[0]+df['purchase'].value_counts()[1]), 4),
                     'features':df.shape[1]}


# In[8]:


print(descriptives_dict)


# In[9]:


##### PREPARE DATA FOR MODELING #####


# In[10]:


y_train = df[df['hit_time_gmt'] <= '2016-05-10 23:59:59']['purchase']
X_train = df[df['hit_time_gmt'] <= '2016-05-10 23:59:59'].copy()
X_train.drop(['purchase', 'hit_time_gmt', 'visitor_id'], axis=1, inplace=True)


# In[11]:


y_test = df[df['hit_time_gmt'] > '2016-05-10 23:59:59']['purchase']
X_test = df[df['hit_time_gmt'] > '2016-05-10 23:59:59'].copy()
X_test.drop(['purchase', 'hit_time_gmt', 'visitor_id'], axis=1, inplace=True)


# In[12]:


##### TRAIN AND TEST MODELS #####


# In[13]:


### import libraries needed for modeling and performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report


# In[14]:


lr = LogisticRegression()


# In[15]:


kfold = KFold(n_splits=10, random_state=42)
cv_results = cross_val_score(lr, X_train, y_train, cv=kfold, scoring='accuracy')
print(cv_results)
print(cv_results.mean())
print(cv_results.std())


# In[16]:


lr.fit(X_train, y_train)


# In[17]:


y_pred = lr.predict(X_test)


# In[18]:


##### EVALUATE MODELS #####


# In[19]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
print('\n')
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print('AUC: %.2f' % auc(fpr, tpr))
print('\n')
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred))
print('\n')
print('Classification report')
print(classification_report(y_test, y_pred))

# true negatives C[0,0] false negatives C[1,0] true positives C[1,1] false positives is C[0,1]


# In[20]:


print('Total execution time: ', datetime.now() - start_time)

