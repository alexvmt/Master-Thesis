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

from helper_functions import *



# input file
input_file = 'clickstream_0516-1016_processed_final.pkl.gz'

# load data
print('Starting loading data...')

df = pd.read_pickle('../data/processed_data/'+input_file, compression='gzip')

print('Loading data complete.')



### PREPROCESSING
print('Starting preprocessing...')

# select sample size aka number of unique visitor ids
with open('../data/processed_data/unique_visitor_ids.pkl', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
#sample_size = 125000
#sample_size = 250000
sample_size = 500000
#sample_size = 1000000
#sample_size = 2000000
unique_visitor_ids_to_sample = unique_visitor_ids[:sample_size]
df = df[df['visitor_id'].isin(unique_visitor_ids_to_sample)]

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
#target = 'purchase_within_current_visit'
#target = 'purchase_within_next_24_hours'
#target = 'purchase_within_next_7_days'
target = 'purchase_within_next_visit'
targets.remove(target)
df.drop(targets, axis=1, inplace=True)

# split sample in stratified training and test sets
from sklearn.model_selection import train_test_split
y = df[target]
X = df.drop(target, axis=1)
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=random_state)

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
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print('Preprocessing complete.')



### DESCRIPTIVES
print('Starting calculating descriptives...')

# calculate descriptives
descriptives_dict = {'unique visitors sample' : df['visitor_id'].nunique(),
                     'visits sample' : df.shape[0],
                     'features sample' : df.shape[1]-3,
                     'conversion rate sample' : round(df[target].sum()/df.shape[0], 4),
                     'days sample' : len(df['visit_start_time_gmt'].apply(lambda x: x.date()).unique()),
                     'unique visitors X_train' : X_train['visitor_id'].nunique(),
                     'visits X_train' : X_train.shape[0],
                     'features X_train' : X_train.shape[1]-2,
                     'conversion rate X_train' : round(y_train.sum()/X_train.shape[0], 4),
                     'days X_train' : len(X_train['visit_start_time_gmt'].apply(lambda x: x.date()).unique()),
                     'unique visitors X_test' : X_test['visitor_id'].nunique(),
                     'visits X_test' : X_test.shape[0],
                     'features X_test' : X_test.shape[1]-2,
                     'conversion rate X_test' : round(y_test.sum()/X_test.shape[0], 4),
                     'days X_test' : len(X_test['visit_start_time_gmt'].apply(lambda x: x.date()).unique())}

# save descriptives
save_descriptives('../results/descriptives/sample_descriptives_'+str(sample_size)+'_unique_visitors_'+target+'.txt', descriptives_dict)

# drop visitor_id and visit_start_time_gmt from training and test sets
X_train.drop(['visitor_id', 'visit_start_time_gmt'], axis=1, inplace=True)
X_test.drop(['visitor_id', 'visit_start_time_gmt'], axis=1, inplace=True)

print('Calculating descriptives complete.')

# save sample
print('Starting saving sample...')

df.to_pickle('../data/processed_data/sample_'+str(sample_size)+'_unique_visitors_'+target+'.pkl', compression='gzip')

print('Saving sample complete.')



### MODELING
print('Start training, testing and evaluating models...')

# import models and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report

# build models and set up dictionary for saving model output
models = []
models.append(('LR', LogisticRegression(random_state=random_state)))
models.append(('DT', DecisionTreeClassifier(random_state=random_state)))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(random_state=random_state)))
models.append(('SVM', SVC(random_state=random_state)))
models.append(('BOOST', GradientBoostingClassifier(random_state=random_state)))
models.append(('BAG', BaggingClassifier(random_state=random_state)))
models.append(('NN1', MLPClassifier(random_state=random_state, hidden_layer_sizes=(X_train.shape[1]))))
models.append(('NN3', MLPClassifier(random_state=random_state, hidden_layer_sizes=(X_train.shape[1],X_train.shape[1],X_train.shape[1]))))
models.append(('NN5', MLPClassifier(random_state=random_state, hidden_layer_sizes=(X_train.shape[1],X_train.shape[1],X_train.shape[1],X_train.shape[1],X_train.shape[1]))))

model_output_dict = {}

# train, test and evaluate each model in turn
for name, model in models:

    # train
    print('Training '+name+'...')
	
    training_start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - training_start_time)
	
    print('Training '+name+' complete, training time: '+str(training_time)+'.')

    # test
    print('Testing '+str(name)+'...')
	
    testing_start_time = datetime.now()
    y_pred = model.predict(X_test)
    testing_time = datetime.now() - testing_start_time
	
    print('Testing '+name+' complete, test time: '+str(testing_time)+'.')

    # evaluate
    print('Evaluating '+name+'...')
	
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
    model_output_dict['training_time'+name] = str(training_time)
    model_output_dict['testing_time'+name] = str(testing_time)
    model_output_dict['accuracy'+name] = accuracy
    model_output_dict['auc'+name] = auc_score
    model_output_dict['true_negatives'+name] = true_negatives
    model_output_dict['false_negatives'+name] = false_negatives
    model_output_dict['true_positives'+name] = true_positives
    model_output_dict['false_positives'+name] = false_positives
    model_output_dict['precision'+name] = precision
    model_output_dict['recall'+name] = recall
    model_output_dict['f1_score'+name] = f1_score
	
    print('Evaluating '+name+' complete.')
    
    # calculate feature importance for DT, RF and BOOST
    print('Starting calculating feature importance...')
	
    if name in ['DT', 'RF', 'BOOST']:
        feature_importance = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), X_train.columns), reverse=True)
        save_descriptives('../results/models/feature_importance_'+name+'_'+str(sample_size)+'_unique_visitors_'+target+'.txt', feature_importance)
		
    print('Calculating feature importance complete.')

# save model output
save_descriptives('../results/models/model_output_'+str(sample_size)+'_unique_visitors_'+target+'.txt', model_output_dict)

print('Training, testing and evaluating models complete.')



# calculate additional feature importance with ExtraTreeClassifier
print('Starting calculating feature importance with ExtraTreeClassifier...')

from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=1000, random_state=random_state)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
feature_importance = []
for f in range(X_train.shape[1]):
	feature_importance.append("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
save_descriptives('../results/models/feature_importance_ExtraTreeClassifier'+str(sample_size)+'_unique_visitors_'+target+'.txt', feature_importance)

print('Calculating feature importance with ExtraTreeClassifier complete.')

	

print('Modeling complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time
save_descriptives('../results/descriptives/modeling_run_time.txt', run_time)