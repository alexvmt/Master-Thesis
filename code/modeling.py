#!/usr/bin/env python
# coding: utf-8

##### MODELING #####

print('Starting modeling...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd
import pickle



# input file
input_file = 'clickstream_0516-1016_processed_final.pkl.gz'

# load data
print('Starting loading data...')

df = pd.read_pickle('../data/processed_data/'+input_file, compression='gzip')

print('Loading data complete.')



### PREPROCESSING
print('Starting preprocessing...')

# select sample size aka number of unique visitor ids
with open('../data/processed_data/unique_visitor_ids.plk', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
sample_size = 125000
unique_visitor_ids_to_sample = unique_visitor_ids[:sample_size]
df = df[df['visitor_id'] in unique_visitor_ids_to_sample]

# drop categorical columns either aggregated by first or by last occurrence
#first_or_last = '_first'
first_or_last = '_last'
for column in df.columns:
    if first_or_last in column:
	    df.drop(column, axis=1, inplace=True)
    else:
	    pass
		
# select target
targets = ['purchase_within_current_visit', 
'purchase_within_next_24_hours', 
'purchase_within_next_7_days', 
'purchase_within_next_visit']
target = 'purchase_within_current_visit'
#target = 'purchase_within_next_24_hours'
#target = 'purchase_within_next_7_days'
#target = 'purchase_within_next_visit'
targets.remove(target)
df.drop(targets, axis=1, inplace=True)

# standardize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = ['visit_page_num_max',
'product_view_boolean_sum',
'checkout_boolean_sum',
'cart_addition_boolean_sum',
'cart_removal_boolean_sum',
'cart_view_boolean_sum',
'campaign_view_boolean_sum',
'cart_value_sum',
'page_view_boolean_sum',
'last_purchase_num_max',
'product_items_sum',
'product_item_price_sum',
'standard_search_results_clicked_sum',
'standard_search_started_sum',
'suggested_search_results_clicked_sum']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, target, stratify=target, test_size=0.25)

print('Preprocessing complete.')



### DESCRIPTIVES
print('Starting calculting descriptives...')

descriptives_dict = {'unique visitors total' : df['visitor_id'].nunique(),
                     'visits total' : df.shape[0],
                     'features total' : df.shape[1]-3,
                     'conversion rate total' : round(df[target].sum()/df.shape[0], 4),
                     'days total' : len(df['visit_start_time_gmt'].apply(lambda x: x.date()).unique()),
                     'unique visitors X_train' : X_train['visitor_id'].nunique(),
                     'visits X_train' : X_train.shape[0],
                     'features X_train' : X_train.shape[1]-2,
                     'conversion rate X_train' : round(y_train[target].sum()/X_train.shape[0], 4),
                     'days X_train' : len(X_train['visit_start_time_gmt'].apply(lambda x: x.date()).unique()),
                     'unique visitors X_test' : X_test['visitor_id'].nunique(),
                     'visits X_test' : X_test.shape[0],
                     'features X_test' : df.shape[1]-2,
                     'conversion rate X_test' : round(y_test[target].sum()/X_test.shape[0], 4),
                     'days X_test' : len(X_test['visit_start_time_gmt'].apply(lambda x: x.date()).unique())}

f = open('../data/descriptives/sample_descriptives_'+str(sample_size)+'_unique_visitors_'+target+'.txt', 'w')
f.write(str(descriptives_dict))
f.close()

X_train.drop(['visitor_id', 'visit_start_time_gmt'], axis=1, inplace=True)
X_test.drop(['visitor_id', 'visit_start_time_gmt'], axis=1, inplace=True)

print('Calculating descriptives complete.')



### MODELING
print('Start training, testing and evaluating models...')

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

model_output_dict = {}

# train, test and evaluate each model in turn
for name, model in models:
    
    # train
    print('Training ', name, '...')
    training_start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - training_start_time)
    print('Training ', name, 'complete, training time: ', training_time, '.')
    
    # test
    print('Testing ', name, '...')
    testing_start_time = datetime.now()
    y_pred = model.predict(X_test)
    testing_time = datetime.now() - testing_start_time
    print('Testing ', name, 'complete, test time: ', testing_time, '.')
    
    # evaluate
    print('Evaluating ', name, '...')
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    true_negatives = confusion_matrix(y_test, y_pred)[0,0]
    false_negatives = confusion_matrix(y_test, y_pred)[1,0]
    true_positives = confusion_matrix(y_test, y_pred)[1,1]
    false_positives = confusion_matrix(y_test, y_pred)[0,1]
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    f1_score = 2*((precision*recall)/(precision+recall))
	
	# add model output to model_output_dict
    model_output_dict['training_time'+str(name)] = training_time
    model_output_dict['testing_time'+str(name)] = testing_time
    model_output_dict['accuracy'+str(name)] = accuracy
    model_output_dict['auc'+str(name)] = auc
    model_output_dict['true_negatives'+str(name)] = true_negatives
    model_output_dict['false_negatives'+str(name)] = false_negatives
    model_output_dict['true_positives'+str(name)] = true_positives
    model_output_dict['false_positives'+str(name)] = false_positives
    model_output_dict['precision'+str(name)] = precision
    model_output_dict['recall'+str(name)] = recall
    model_output_dict['f1_score'+str(name)] = f1_score

    print('Evaluating ', name, 'complete.')

# save model output
f = open('../data/model_output/model_output_'+str(sample_size)+'_unique_visitors_'+target+'.txt', 'w')
f.write(str(lr_output_dict))
f.close()

print('Training, testing and evaluating models complete.')



### SMOTE



### HYPERPARAMETER TUNING



# save run time
print('Modeling complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'modeling_run_time.txt'
run_time_dict = {'modeling run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)