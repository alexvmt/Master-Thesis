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

# select data for feature selection
with open('../data/processed_data/unique_visitor_ids.pkl', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
df = df[df['visitor_id'].isin(unique_visitor_ids[2000001:])]
print('All features: ', df.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours'], axis=1).shape[1])

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
scaler.fit(df[numerical_features])
df[numerical_features] = scaler.transform(df[numerical_features])

print('Preprocessing complete.')



### FEATURE SELECTION
print('Starting feature selection...')

# select k best features based on independent sample of 453174 unique visitors that are not contained in any of the training and test sets

# check whether selected sample has a conversion rate similar to other samples
print('conversion rate: ', round(sum(df['purchase_within_next_24_hours'])/len(df['purchase_within_next_24_hours'])*100, 4), '%')

# split df in target and features
y = df['purchase_within_next_24_hours']
X = df.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours'], axis=1)

# select best features using ANOVA F-test and p-values
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k='all')
X_best = selector.fit_transform(X, y)
features = X.columns.values[selector.get_support()]
f_values = selector.scores_[selector.get_support()]
p_values = selector.pvalues_[selector.get_support()]
best_features = pd.DataFrame({'Features': features, 'F-values': f_values, 'p-values': p_values})

# save all features and select only features are statistically significant at the one percent level
best_features.sort_values('F-values', ascending=False).set_index('Features').to_pickle('../results/descriptives/best_features.pkl.gz', compression='gzip')
k_best_features = list(best_features[best_features['p-values'] <= 0.01]['Features'])
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
save_script_run_time('../results/descriptives/feature_selection_run_time.txt', run_time)
