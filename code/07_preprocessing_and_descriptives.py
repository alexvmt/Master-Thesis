#!/usr/bin/env python
# coding: utf-8

##### PREPROCESSING AND DESCRIPTIVES #####

print('Starting preprocessing and calculating descriptives...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import sys
params = sys.argv

import numpy as np
import pandas as pd
import pickle

from helper_functions import *

# print parameters passed to script
print('Training set size:', params[1])
print('Categorical aggregation mode:', params[2])



### LOAD DATA
print('Starting loading data...')

# input file
input_file = 'clickstream_0516-1016_prepared.pkl.gz'

# output files
output_file_train = 'train_'+params[1]+'.pkl.gz'
output_file_test = 'test_'+params[1]+'.pkl.gz'
output_file_descriptives = 'descriptives_'+params[1]+'.pkl.gz'

# load data
df = pd.read_pickle('../data/processed_data/'+input_file)

print('Loading data complete.')



### PREPROCESSING
print('Starting preprocessing...')

# create training set (80%)
with open('../data/processed_data/unique_visitor_ids.pkl', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
train = df[df['visitor_id'].isin(unique_visitor_ids[:(int(params[1]))])]

# create test set (20%)
test = df[df['visitor_id'].isin(unique_visitor_ids[(int(params[1])+1):(int(params[1])+1+(int(int(params[1])/0.8*0.2)))])]

# drop categorical features that do not match the aggregation mode set via params[2]
if params[2] == 'first':
    categorical_features_to_drop = [column for column in train.columns if ('_last' in column) & ('_in_last_' not in column)]
    train.drop(categorical_features_to_drop, axis=1, inplace=True)
    test.drop(categorical_features_to_drop, axis=1, inplace=True)
elif params[2] == 'last':
    categorical_features_to_drop = [column for column in train.columns if '_first' in column]
    train.drop(categorical_features_to_drop, axis=1, inplace=True)
    test.drop(categorical_features_to_drop, axis=1, inplace=True)
else:
    print('Aggregation mode for categorical features not found. Please select one from the available options: first or last.')

# standardize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = ['visit_page_num_max',
'product_view_boolean_sum',
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
scaler.fit(train[numerical_features])
train[numerical_features] = scaler.transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])

# keep only k best features
with open('../data/processed_data/k_best_features.pkl', 'rb') as f:
   k_best_features = pickle.load(f)
for column in train.columns:
    if column in ['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days']:
        pass
    elif column not in k_best_features:
        train.drop(column, axis=1, inplace=True)
        test.drop(column, axis=1, inplace=True)
    else:
        pass

# save training and test sets
train.to_pickle('../data/training_and_test_sets/'+output_file_train, compression='gzip')
test.to_pickle('../data/training_and_test_sets/'+output_file_test, compression='gzip')

print('Preprocessing complete.')



### DESCRIPTIVES
print('Starting calculating descriptives...')

# set up dataframe for descriptives
columns = ['unique_visitors',
'visits',
'visitors_with_2_or_more_visits',
'visitors_with_5_or_more_visits',
'unique_days',
'features',
'conversions_24_hours',
'conversion_rate_24_hours',
'conversions_7_days',
'conversion_rate_7_days']
index = ['train_'+params[1], 'test_'+params[1]]
descriptives = pd.DataFrame(index=index, columns=columns)

# calculate descriptives for training set
descriptives.at['train_'+params[1], 'unique_visitors'] = train['visitor_id'].nunique()
descriptives.at['train_'+params[1], 'visits'] = train.shape[0]
visits_per_visitor_train = train[['visitor_id']].groupby('visitor_id').size().reset_index(name='visits')
descriptives.at['train_'+params[1], 'visitors_with_2_or_more_visits'] = visits_per_visitor_train[visits_per_visitor_train['visits'] >= 2].shape[0]
descriptives.at['train_'+params[1], 'visitors_with_5_or_more_visits'] = visits_per_visitor_train[visits_per_visitor_train['visits'] >= 5].shape[0]
descriptives.at['train_'+params[1], 'unique_days'] = len(train['visit_start_time_gmt'].apply(lambda x: x.date()).unique())
descriptives.at['train_'+params[1], 'features'] = train.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1).shape[1]
descriptives.at['train_'+params[1], 'conversions_24_hours'] = train['purchase_within_next_24_hours'].sum()
descriptives.at['train_'+params[1], 'conversion_rate_24_hours'] = round(train['purchase_within_next_24_hours'].sum()/train.shape[0], 4)
descriptives.at['train_'+params[1], 'conversions_7_days'] = train['purchase_within_next_7_days'].sum()
descriptives.at['train_'+params[1], 'conversion_rate_7_days'] = round(train['purchase_within_next_7_days'].sum()/train.shape[0], 4)

# calculate descriptives for test set
descriptives.at['test_'+params[1], 'unique_visitors'] = test['visitor_id'].nunique()
descriptives.at['test_'+params[1], 'visits'] = test.shape[0]
visits_per_visitor_test = test[['visitor_id']].groupby('visitor_id').size().reset_index(name='visits')
descriptives.at['test_'+params[1], 'visitors_with_2_or_more_visits'] = visits_per_visitor_test[visits_per_visitor_test['visits'] >= 2].shape[0]
descriptives.at['test_'+params[1], 'visitors_with_5_or_more_visits'] = visits_per_visitor_test[visits_per_visitor_test['visits'] >= 5].shape[0]
descriptives.at['test_'+params[1], 'unique_days'] = len(test['visit_start_time_gmt'].apply(lambda x: x.date()).unique())
descriptives.at['test_'+params[1], 'features'] = test.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1).shape[1]
descriptives.at['test_'+params[1], 'conversions_24_hours'] = test['purchase_within_next_24_hours'].sum()
descriptives.at['test_'+params[1], 'conversion_rate_24_hours'] = round(test['purchase_within_next_24_hours'].sum()/test.shape[0], 4)
descriptives.at['test_'+params[1], 'conversions_7_days'] = test['purchase_within_next_7_days'].sum()
descriptives.at['test_'+params[1], 'conversion_rate_7_days'] = round(test['purchase_within_next_7_days'].sum()/test.shape[0], 4)

# save descriptives
descriptives.to_pickle('../results/descriptives/'+output_file_descriptives, compression='gzip')

print('Calculating descriptives complete.')



print('Preprocessing and calculating descriptives complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/preprocessing_and_descriptives_run_time_'+params[1]+'.txt', run_time)