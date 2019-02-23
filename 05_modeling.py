#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### MODELING #####


# In[ ]:


print('Starting modeling...')


# In[ ]:


### import libraries
import numpy as np
import pandas as pd
from datetime import datetime,date

start_time = datetime.now()
print('Start time: ', start_time)


# In[ ]:


#### SELECT INPUT FILE


# In[ ]:


#input_file = '3_day_sample_preprocessed.tsv.gz'
input_file = '3_day_sample_preprocessed_with_additional_features.tsv.gz'
#input_file = '6_week_sample_preprocessed.tsv.gz'
#input_file = '6_week_sample_preprocessed_with_additional_features.tsv.gz'
#input_file = '12_week_sample_preprocessed.tsv.gz'
#input_file = '12_week_sample_preprocessed_with_additional_features.tsv.gz'
#input_file = '25_week_sample_preprocessed.tsv.gz'
#input_file = '25_week_sample_preprocessed_with_additional_features.tsv.gz'

print('Input file selected: ', input_file)


# In[ ]:


##### LOAD DATA
print('Loading data...')


# In[ ]:


df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt', 'date_time'])

print('Loading data complete.')


# In[ ]:


##### DESCRIPTIVES
print('Calculating descriptives...')


# In[ ]:


descriptives_dict = {'unique visitors' : df['visitor_id'].nunique(),
                     'visits' : df.shape[0],
                     'conversion rate' : round(df['purchase'].value_counts()[1]/(len(df['purchase'])), 4),
                     'features' : df.shape[1]}
print('Sample descriptives: ', descriptives_dict)

print('Calculating descriptives complete.')


# In[ ]:


##### PREPARE DATA FOR MODELING


# In[ ]:


print('Preparing data for modeling...')


# In[ ]:


# drop columns that have many missing values, are static or where their usefulness is unclear
cols_to_drop = ['visitor_id_lag', 
                'last_hit_time_gmt_visit', 
                'last_hit_time_gmt_visit_lag',
                'last_date_time_visit',
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


# In[ ]:


### generate training set from 2/3 of the data
y_train = df[df['date_time'] <= '2016-05-10 23:59:59']['purchase']
X_train = df[df['date_time'] <= '2016-05-10 23:59:59'].copy()
X_train = X_train.reset_index(drop=True)


# In[ ]:


train_descriptives_dict = {'unique visitors' : X_train['visitor_id'].nunique(),
                           'visits' : X_train.shape[0],
                           'conversion rate' : round(y_train.value_counts()[1]/(len(y_train)), 4),
                           'features' : df.shape[1] - 4, 
                           'days for training': (X_train['hit_time_gmt'].max() - X_train['hit_time_gmt'].min()).days}
X_train.drop(['purchase', 'hit_time_gmt', 'date_time','visitor_id'], axis=1, inplace=True)
print('Descriptives training set: ', train_descriptives_dict)


# In[ ]:


### generate test set from 1/3 of the data
y_test = df[df['date_time'] > '2016-05-10 23:59:59']['purchase']
X_test = df[df['date_time'] > '2016-05-10 23:59:59'].copy()
X_test = X_test.reset_index(drop=True)


# In[ ]:


test_descriptives_dict = {'unique visitors' : X_test['visitor_id'].nunique(),
                          'visits' : X_test.shape[0],
                          'conversions rate' : round(y_test.value_counts()[1]/(len(y_test)), 4),
                          'features' : X_test.shape[1] - 4, 
                          'days for testing': (X_test['hit_time_gmt'].max() - X_test['hit_time_gmt'].min()).days}
X_test.drop(['purchase', 'hit_time_gmt', 'date_time', 'visitor_id'], axis=1, inplace=True)
print('Descriptives test set: ', test_descriptives_dict)


# In[ ]:


print('Preparing data for modeling complete.')


# In[ ]:


##### TRAIN, TEST AND EVALUATE MODELS
print('Starting training, testing and evaluating models...')


# In[ ]:


### import libraries for modeling and performance evaluation
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report


# In[ ]:


### build models, do 10-fold cross validation and evaluate each model in turn
models = []
models.append(('LR', LogisticRegression()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
#models.append(('RF', RandomForestClassifier()))


# In[ ]:


### test and evaluate each model in turn
for name, model in models:
    
    print('Training ', name, '...')
    training_start_time = datetime.now()
    model.fit(X_train, y_train)
    training_duration = (datetime.now() - training_start_time)
    print('Training ', name, 'complete, training_duration: ', training_duration)
    
    print('Testing ', name, '...')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_duration = datetime.now() - test_start_time
    print('Testing ', name, 'complete, test_duration: ', test_duration)
    
    print('Evaluating ', name, '...')
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
    print('Evaluating ', name, 'complete.')


# In[ ]:


##### SMOTE FOR NOMINAL AND CONTINUOUS DATA


# In[ ]:


print('Starting resampling...')


# In[ ]:


from imblearn.over_sampling import SMOTENC


# In[ ]:


sm = SMOTENC(random_state=42, categorical_features=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                                    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                                    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                                                    71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 85, 86])


# In[ ]:


# resampling
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# In[ ]:


print('old training set size', X_train.shape[0])
print('new training set size', X_train_res.shape[0])
print('old conversion rate', round(y_train.value_counts()[1]/(len(y_train)), 4))
print('new conversion rate', round(y_train_res.sum()/(len(y_train_res)), 4))


# In[ ]:


print('Resampling complete.')


# In[ ]:


print('Reevaluation models...')


# In[ ]:


### test and evaluate each model in turn using resampled data
for name, model in models:
    
    print('Training ', name, '...')
    training_start_time = datetime.now()
    model.fit(X_train_res, y_train_res)
    training_duration = (datetime.now() - training_start_time)
    print('Training ', name, 'complete, training_duration: ', training_duration)
    
    print('Testing ', name, '...')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_duration = datetime.now() - test_start_time
    print('Testing ', name, 'complete, test_duration: ', test_duration)
    
    print('Evaluating ', name, '...')
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
    print('Evaluating ', name, 'complete.')


# In[ ]:


print('Reevaluating models complete.')


# In[ ]:


print('Modeling complete.')
print('Run time: ', datetime.now() - start_time)

