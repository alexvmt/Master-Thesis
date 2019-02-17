#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### MODELING #####


# In[2]:


print('Starting modeling...')


# In[3]:


### import libraries
import numpy as np
import pandas as pd
from datetime import datetime,date

start_time = datetime.now()
print('Start time: ', start_time)


# In[4]:


#### SELECT INPUT FILE


# In[5]:


#input_file = '3_day_sample_preprocessed.tsv.gz'
input_file = '3_day_sample_preprocessed_with_additional_features.tsv.gz'
#input_data = '6_week_sample_preprocessed.tsv.gz'
#input_file = '6_week_sample_preprocessed_with_additional_features.tsv.gz'
#input_data = '12_week_sample_preprocessed.tsv.gz'
#input_file = '12_week_sample_preprocessed_with_additional_features.tsv.gz'
#input_data = '25_week_sample_preprocessed.tsv.gz'
#input_file = '25_week_sample_preprocessed_with_additional_features.tsv.gz'

print('Input file selected: ', input_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt'])

print('Loading data complete.')


# In[8]:


##### DESCRIPTIVES
print('Calculating descriptives...')


# In[9]:


descriptives_dict = {'unique_visitors' : df['visitor_id'].nunique(),
                     'visits' : df.shape[0],
                     'percentage_purchases' : round(df['purchase'].value_counts()[1]/(len(df['purchase'])), 4),
                     'features' : df.shape[1]}
print('Sample descriptives: ', descriptives_dict)

print('Calculating descriptives complete.')


# In[10]:


##### PREPARE DATA FOR MODELING #####


# In[11]:


print('Preparing data for modeling...')


# In[12]:


# drop columns that have many missing values, are static or where their usefulness is unclear
cols_to_drop = ['visitor_id_lag', 
                'last_hit_time_gmt_visit', 
                'last_hit_time_gmt_visit_lag',
                'days_since_last_visit',
                'purchase_date',
                'purchase_date_lag',
                'days_since_last_purchase',
                'country', 
                'geo_region',
                'geo_city',
                'geo_zip',
                'geo_dma',
                'post_channel',
                'search_page_num',
                'net_promoter_score_raw_(v10)_-_user',
                'registration_(any_form)_(e20)',
                'hit_of_logged_in_user_(e23)', # duplicate of login_status
                'newsletter_signup_(any_form)_(e26)', 
                'newsletter_subscriber_(e27)', 
                'user_gender_(v61)',
                'user_age_(v62)',
                'login_success_(e72)', 
                'logout_success_(e73)', 
                'login_fail_(e74)', 
                'registration_fail_(e75)',
                'product_categories_level_1',
                'product_categories_level_2',
                'product_categories_level_3']

for col in cols_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
    else:
        pass


# In[13]:


### 2 days for training
y_train = df[df['hit_time_gmt'] <= '2016-05-10 23:59:59']['purchase']
X_train = df[df['hit_time_gmt'] <= '2016-05-10 23:59:59'].copy()
X_train = X_train.reset_index(drop=True)


# In[14]:


train_descriptives_dict = {'unique_visitors' : X_train['visitor_id'].nunique(),
                           'visits' : X_train.shape[0],
                           'percentage_purchases' : round(y_train.value_counts()[1]/(len(y_train)), 4),
                           'features' : df.shape[1], 
                           'days_for_training': (X_train['hit_time_gmt'].max() - X_train['hit_time_gmt'].min()).days}
X_train.drop(['purchase', 'hit_time_gmt', 'visitor_id'], axis=1, inplace=True)
print('Descriptives training set: ', train_descriptives_dict)


# In[15]:


### 1 day for testing
y_test = df[df['hit_time_gmt'] > '2016-05-10 23:59:59']['purchase']
X_test = df[df['hit_time_gmt'] > '2016-05-10 23:59:59'].copy()


# In[16]:


test_descriptives_dict = {'unique_visitors' : X_test['visitor_id'].nunique(),
                          'visits' : X_test.shape[0],
                          'percentage_purchases' : round(y_test.value_counts()[1]/(len(y_test)), 4),
                          'features' : X_test.shape[1], 
                          'days_for_training': (X_test['hit_time_gmt'].max() - X_test['hit_time_gmt'].min()).days}
X_test.drop(['purchase', 'hit_time_gmt', 'visitor_id'], axis=1, inplace=True)
print('Descriptives test set: ', test_descriptives_dict)


# In[17]:


print('Preparing data for modeling complete.')


# In[18]:


##### TRAIN AND TEST MODELS #####
print('Training models...')


# In[19]:


### import libraries for modeling and performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report


# In[21]:


### build models, do 10-fold cross validation and evaluate each model in turn
models = []
models.append(('LR', LogisticRegression()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
#models.append(('RF', RandomForestClassifier()))

results = []
names = []

for name, model in models:
    
    cv_start_time = datetime.now()
    kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_duration = datetime.now() - cv_start_time
    
    results.append(cv_results)
    names.append(name)
    
    cv_msg = '%s: %f (mean accuracy) %f (standard deviation) %s (cv duration)' % (name, cv_results.mean(), cv_results.std(), cv_duration)
    print(cv_msg)


# In[22]:


### train each model in turn
for name, model in models:
    
    training_start_time = datetime.now()
    model.fit(X_train, y_train)
    training_duration = (datetime.now() - training_start_time)
    
    print(name, ': ', training_duration, '(training_duration)')
    
print('Training models complete.')


# In[23]:


##### EVALUATE MODELS #####
print('Starting evaluating models...')


# In[24]:


### test and evaluate each model in turn
for name, model in models:
    
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_duration = datetime.now() - test_start_time
    
    print(name, ': ', test_duration, '(test_duration)')
    
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f%%' % (accuracy * 100.0))
    print('\n')
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print('AUC: %.2f' % auc(fpr, tpr))
    print('\n')
    
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('true negatives C[0,0] false negatives C[1,0] true positives C[1,1] false positives is C[0,1]')
    print('\n')
    
    print('Classification report')
    print(classification_report(y_test, y_pred))


# In[25]:


print('Modeling and evaluation complete.')
print('Run time: ', datetime.now() - start_time)

