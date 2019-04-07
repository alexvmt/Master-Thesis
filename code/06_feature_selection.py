#!/usr/bin/env python
# coding: utf-8

##### FEATURE SELECTION #####

print('Starting feature selection...')

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

# output file
output_file = 'k_best_features.pkl'

# load data
df = pd.read_pickle('../data/processed_data/'+input_file)

print('Loading data complete.')



### PREPROCESSING
print('Starting preprocessing...')

# create training set (80%)
with open('../data/processed_data/unique_visitor_ids.pkl', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
train = df[df['visitor_id'].isin(unique_visitor_ids[:(int(params[1]))])]
print('All features: ', train.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1).shape[1])

# drop categorical features that do not match the aggregation mode set via params[2]
if params[2] == 'first':
    categorical_features_to_drop = [column for column in train.columns if ('_last' in column) & ('_in_last_' not in column)]
    train.drop(categorical_features_to_drop, axis=1, inplace=True)
elif params[2] == 'last':
    categorical_features_to_drop = [column for column in train.columns if '_first' in column]
    train.drop(categorical_features_to_drop, axis=1, inplace=True)
else:
    print('Aggregation mode for categorical features not found. Please select one from the available options: first or last.')
print('Features without categorical features aggregated by', params[2], ' occurrence: ', train.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1).shape[1])

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

print('Preprocessing complete.')



### FEATURE SELECTION
print('Starting feature selection...')

# select k best features based on 100000 unique visitors training set since it is the only data that is not part of the test sets of the larger samples
# e.g. 200000 unique visitor training set contains the test set used to evaluate the 100000 unique visitor training set
# training and test sets have to overlap in order to allow for better comparison of samples of different size to measure the effect of sample size

# split training set in features and targets
y_train_24 = train['purchase_within_next_24_hours']
y_train_7 = train['purchase_within_next_7_days']
X_train = train.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1)

# calculate feature importance using ANOVA, F test and p values for target == purchase_within_next_24_hours
from sklearn.feature_selection import SelectKBest, f_classif
selector_24 = SelectKBest(f_classif, k='all')
X_train_k_best_24 = selector_24.fit_transform(X_train, y_train_24)
features_24 = X_train.columns.values[selector_24.get_support()]
scores_24 = selector_24.scores_[selector_24.get_support()]
p_values_24 = selector_24.pvalues_[selector_24.get_support()]
k_best_24 = pd.DataFrame({'features': features_24, 'scores_24': scores_24, 'p_values_24': p_values_24})

# calculate feature importance using ANOVA, F test and p values for target == purchase_within_next_7_days
selector_7 = SelectKBest(f_classif, k='all')
X_train_k_best_7 = selector_7.fit_transform(X_train, y_train_7)
features_7 = X_train.columns.values[selector_7.get_support()]
scores_7 = selector_7.scores_[selector_7.get_support()]
p_values_7 = selector_7.pvalues_[selector_7.get_support()]
k_best_7 = pd.DataFrame({'features': features_7, 'scores_7': scores_7, 'p_values_7': p_values_7})

# merge scores and p values of calcuations for purchase_within_next_24_hours and purchase_within_next_7_days
k_best = pd.merge(k_best_24, k_best_7, on='features')

# select only features that have p values of <= 0.01 for both purchase_within_next_24_hours and purchase_within_next_7_days
k_best_features = list(k_best[(k_best['p_values_24'] <= 0.01) & (k_best['p_values_7'] <= 0.01)]['features'])
print('k best features: ', len(k_best_features))

print('Feature selection complete.')



### WRITE DATA
print('Starting writing data...')

with open('../data/processed_data/'+output_file, 'wb') as f:
   pickle.dump(k_best_features, f)

print('Writing data complete.')



print('Feature selection complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/feature_selection_run_time_'+params[1]+'.txt', run_time)