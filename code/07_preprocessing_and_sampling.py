#!/usr/bin/env python
# coding: utf-8

##### PREPROCESSING AND SAMPLING #####

print('Starting preprocessing and sampling...')

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

# print selected training set size
print('Training set size:', params[1])



### LOAD DATA
print('Starting loading data...')

# input file
input_file = 'clickstream_0516-1016_prepared.pkl.gz'

# output files
output_file_train = 'train_'+params[1]+'.pkl.gz'
output_file_test = 'test_'+params[1]+'.pkl.gz'
output_file_descriptives = 'preprocessing_and_sampling_descriptives_'+str(params[1])+'.pkl.gz'

# load data
df = pd.read_pickle('../data/processed_data/'+input_file)

# set up dataframe for descriptives
columns = ['rows_pre',
'rows_post',
'columns_pre',
'columns_post',
'unique_visitors_pre',
'unique_visitors_post',
'run_time']
index = [input_file[:16]]
preprocessing_and_sampling_descriptives = pd.DataFrame(index=index, columns=columns)

# save pre descriptives
preprocessing_and_sampling_descriptives['rows_pre'] = df.shape[0]
preprocessing_and_sampling_descriptives['columns_pre'] = df.shape[1]
preprocessing_and_sampling_descriptives['unique_visitors_pre'] = df['visitor_id'].nunique()

print('Loading data complete.')



### PREPROCESSING
print('Starting preprocessing and sampling...')

# create training set (80%)
with open('../data/processed_data/unique_visitor_ids.pkl', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
train = df[df['visitor_id'].isin(unique_visitor_ids[:(int(params[1]))])]

# create test set (20%)
test = df[df['visitor_id'].isin(unique_visitor_ids[(int(params[1])+1):(int(params[1])+1+(int(int(params[1])/0.8*0.2)))])]

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
'suggested_search_results_clicked_sum',
'visit_duration_seconds']
scaler.fit(train[numerical_features])
train[numerical_features] = scaler.transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])

# keep only k best features
with open('../data/processed_data/k_best_features.pkl', 'rb') as f:
   k_best_features = pickle.load(f)
for column in train.columns:
    if column in ['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours']:
        pass
    elif column not in k_best_features:
        train.drop(column, axis=1, inplace=True)
        test.drop(column, axis=1, inplace=True)
    else:
        pass

print('Preprocessing and sampling complete.')



### WRITE DATA
print('Starting writing data...')

# save post descriptives
preprocessing_and_sampling_descriptives['rows_post'] = train.shape[0] + test.shape[0]
preprocessing_and_sampling_descriptives['columns_post'] = train.shape[1] + test.shape[1]
preprocessing_and_sampling_descriptives['unique_visitors_post'] = train['visitor_id'].nunique() + test['visitor_id'].nunique()

train.to_pickle('../data/training_and_test_sets/'+output_file_train, compression='gzip')
test.to_pickle('../data/training_and_test_sets/'+output_file_test, compression='gzip')

print('Writing data complete.')



print('Preprocessing and sampling complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time and descriptives dataframe
preprocessing_and_sampling_descriptives['run_time'] = run_time.seconds
preprocessing_and_sampling_descriptives.to_pickle('../results/descriptives/'+output_file_descriptives, compression='gzip')
