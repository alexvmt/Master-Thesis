#!/usr/bin/env python
# coding: utf-8

##### STAGE AND SAMPLE DESCRIPTIVES #####

print('Starting calculating stage and sample descriptives...')

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

# output files
output_file_stage_descriptives = 'stage_descriptives.pkl.gz'
output_file_sample_descriptives = 'sample_descriptives.pkl.gz'

# load data
df = pd.read_pickle('../data/processed_data/'+input_file)

# load unique visitor ids
with open('../data/processed_data/unique_visitor_ids.pkl', 'rb') as f:
   unique_visitor_ids = pickle.load(f)
   
# keep only k best features
with open('../data/processed_data/k_best_features.pkl', 'rb') as f:
   k_best_features = pickle.load(f)
for column in df.columns:
    if column in ['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours']:
        pass
    elif column not in k_best_features:
        df.drop(column, axis=1, inplace=True)
    else:
        pass
	
print('Loading data complete.')



### DESCRIPTIVES

# processing stage descriptives
print('Starting calculating processing stage descriptives...')

# set up dataframe for stage descriptives
columns = ['Rows',
'Columns',
'Unique visitors']
index = ['Raw',
'Cleaning and mapping',
'Aggregation',
'Preparing target and features',
'Feature selection/preprocessing']
stage_descriptives = pd.DataFrame(index=index, columns=columns)

# load descriptives files
input_files_cleaning_and_mapping_descriptives = ['cleaning_and_mapping_descriptives_clickstream_0516.pkl.gz',
'cleaning_and_mapping_descriptives_clickstream_0616.pkl.gz',
'cleaning_and_mapping_descriptives_clickstream_0716.pkl.gz',
'cleaning_and_mapping_descriptives_clickstream_0816.pkl.gz',
'cleaning_and_mapping_descriptives_clickstream_0916.pkl.gz',
'cleaning_and_mapping_descriptives_clickstream_1016.pkl.gz']
cleaning_and_mapping_descriptives = pd.read_pickle('../results/descriptives/'+input_files_cleaning_and_mapping_descriptives[0])
for input_file in input_files_cleaning_and_mapping_descriptives[1:]:
    cleaning_and_mapping_descriptives = cleaning_and_mapping_descriptives.append(pd.read_pickle('../results/descriptives/'+input_file))
aggregation_descriptives = pd.read_pickle('../results/descriptives/aggregation_descriptives.pkl.gz')
preparing_target_and_features_descriptives = pd.read_pickle('../results/descriptives/preparing_target_and_features_descriptives.pkl.gz')

# raw
stage_descriptives.at['Raw', 'Rows'] = cleaning_and_mapping_descriptives['rows_pre'].sum()
stage_descriptives.at['Raw', 'Columns'] = int(cleaning_and_mapping_descriptives['columns_pre'].unique())
stage_descriptives.at['Raw', 'Unique visitors'] = 4717042

# cleaning and mapping
stage_descriptives.at['Cleaning and mapping', 'Rows'] = cleaning_and_mapping_descriptives['rows_post'].sum()
stage_descriptives.at['Cleaning and mapping', 'Columns'] = int(cleaning_and_mapping_descriptives['columns_post'].unique())
stage_descriptives.at['Cleaning and mapping', 'Unique visitors'] = int(aggregation_descriptives['unique_visitors_post'])

# aggregation
stage_descriptives.at['Aggregation', 'Rows'] = int(aggregation_descriptives['rows_post'])
stage_descriptives.at['Aggregation', 'Columns'] = int(aggregation_descriptives['columns_post'])
stage_descriptives.at['Aggregation', 'Unique visitors'] = int(aggregation_descriptives['unique_visitors_post'])

# prepating target and features
stage_descriptives.at['Preparing target and features', 'Rows'] = int(preparing_target_and_features_descriptives['rows_post'])
stage_descriptives.at['Preparing target and features', 'Columns'] = int(preparing_target_and_features_descriptives['columns_post'])
stage_descriptives.at['Preparing target and features', 'Unique visitors'] = int(preparing_target_and_features_descriptives['unique_visitors_post'])

# feature selection/preprocessing
stage_descriptives.at['Feature selection/preprocessing', 'Rows'] = df.shape[0]
stage_descriptives.at['Feature selection/preprocessing', 'Columns'] = df.shape[1]
stage_descriptives.at['Feature selection/preprocessing', 'Unique visitors'] = df['visitor_id'].nunique()

# save processing stage descriptives
stage_descriptives.to_pickle('../results/descriptives/'+output_file_stage_descriptives, compression='gzip')

print('Calculating processing stage descriptives complete.')



# sample descriptives
print('Starting calculating sample descriptives...')

# set up dataframe for sample descriptives
columns = ['Unique visitors',
'Sessions',
'Visitors with >= 2 sessions',
'Visitors with >= 5 sessions',
'Mean # sessions',
'Median # sessions',
'Standard deviation # sessions',
'Buyers',
'Conversions',
'Conversion rate']
index = ['Sample 1',
'Sample 2',
'Sample 3',
'Sample 4',
'Sample 5',
'Sample 6',
'Sample 7',
'Sample 8',
'Sample 9',
'Sample 10',
'Full sample']
sample_descriptives = pd.DataFrame(index=index, columns=columns)

# calculate descriptives for samples with different numbers of unique visitors
unique_visitors_training_set = [3125, 6250, 12500, 25000, 50000, 100000, 200000, 400000, 800000, 1600000, 2453174]
for unique_visitors_training_set,index in zip(unique_visitors_training_set,index):

    if unique_visitors_training_set != 2453174:
        sample_size = int(unique_visitors_training_set+unique_visitors_training_set/0.8*0.2)
    else:
        sample_size = 2453174
		
    print('Sample size: '+str(sample_size))
	
    sample = df[df['visitor_id'].isin(unique_visitor_ids[:sample_size])]
	
    sample_descriptives.at[index, 'Unique visitors'] = sample['visitor_id'].nunique()
    sample_descriptives.at[index, 'Sessions'] = sample.shape[0]
	
    visits_per_visitor = sample[['visitor_id']].groupby('visitor_id').size().reset_index(name='visits')
    sample_descriptives.at[index, 'Visitors with >= 2 sessions'] = visits_per_visitor[visits_per_visitor['visits'] >= 2].shape[0]
    sample_descriptives.at[index, 'Visitors with >= 5 sessions'] = visits_per_visitor[visits_per_visitor['visits'] >= 5].shape[0]
	
    sample_descriptives.at[index, 'Mean # sessions'] = visits_per_visitor['visits'].mean()
    sample_descriptives.at[index, 'Median # sessions'] = visits_per_visitor['visits'].median()
    sample_descriptives.at[index, 'Standard deviation # sessions'] = visits_per_visitor['visits'].std()
	
    buyers = sample[sample['purchase_within_next_24_hours'] == 1]['visitor_id'].unique()
    sample_descriptives.at[index, 'Buyers'] = buyers.shape[0]
    sample_descriptives.at[index, 'Conversions'] = sample['purchase_within_next_24_hours'].sum()
    sample_descriptives.at[index, 'Conversion rate'] = round(sample['purchase_within_next_24_hours'].sum()/len(sample['purchase_within_next_24_hours']), 4)

    print('Sample size '+str(sample_size)+' complete.')

# save sample descriptives
sample_descriptives.to_pickle('../results/descriptives/'+output_file_sample_descriptives, compression='gzip')

print('Calculating sample descriptives complete.')



print('Calculating stage and sample descriptives complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/stage_and_sample_descriptives_run_time.txt', run_time)
