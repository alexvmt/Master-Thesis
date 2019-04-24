#!/usr/bin/env python
# coding: utf-8

##### CLEANING AND MAPPING #####

print('Starting cleaning and mapping...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import sys
params = sys.argv

import numpy as np
import pandas as pd

from helper_functions import *



### LOAD DATA
print('Starting loading data...')

# input file
input_file = params[1]

# output file
output_file = params[1][:17]+'cleaned_and_mapped.pkl.gz'

# load column headers
column_headers = pd.read_csv('../data/mapping_files/column_headers.tsv', sep='\t')

# select columns
columns = ['exclude_hit',
'hit_source',
'connection_type',
'country',
'va_closer_id',
'post_event_list',
'ref_type',
'post_search_engine',
'user_agent',
'product_items',
'product_item_price',
'product_categories',
'post_cookies',
'post_persistent_cookie',
'visit_start_time_gmt',			   
'hit_time_gmt',
'date_time',
'visit_page_num',
'visitor_id',
'visit_num',
'search_page_num',
'new_visit',
'hourly_visitor',
'daily_visitor',
'weekly_visitor',
'monthly_visitor',
'quarterly_visitor',
'yearly_visitor',
'purchase_boolean',
'product_view_boolean',
'checkout_boolean',
'cart_addition_boolean',
'cart_removal_boolean',
'cart_view_boolean',
'campaign_view_boolean',
'page_view_boolean',
'last_purchase_num',
'post_evar10',
'post_evar34',
'post_evar50',
'post_evar61',
'post_evar62']

# load data
df = pd.read_csv('../data/raw_data/'+input_file, compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, names=column_headers, usecols=columns)

# set up dataframe for descriptives
columns = ['rows_pre',
'rows_post',
'columns_pre',
'columns_post',
'unique_visitors_pre',
'unique_visitors_post',
'run_time']
index = [input_file[:16]]
cleaning_and_mapping_descriptives = pd.DataFrame(index=index, columns=columns)

# save pre descriptives
cleaning_and_mapping_descriptives['rows_pre'] = df.shape[0]
cleaning_and_mapping_descriptives['columns_pre'] = df.shape[1]
cleaning_and_mapping_descriptives['unique_visitors_pre'] = df['visitor_id'].nunique()

print('Loading data complete.')



### PROCESS INPUTE FILE
print('Processing '+input_file+'...')

# drop rows
df = drop_rows(df)

# connection type mapping
df = connection_type_mapping(df)

# country mapping
df = country_mapping(df)

# custom evars mapping
df = custom_evars_mapping(df)

# custom marketing channel mapping
df = custom_marketing_channel_mapping(df)

# standard and custom events mapping
df = custom_and_standard_events_mapping(df)

# referrer type mapping
df = referrer_type_mapping(df)

# search engine mapping
df = search_engine_mapping(df)

# user agent mapping
df = user_agent_mapping(df)

# rename columns
df = rename_columns(df)

# drop columns
df = drop_columns(df)

# fill missing and faulty values
df = fill_missing_and_faulty_values(df)

# cast data types
df = cast_data_types(df)

print('Processing '+input_file+' complete.')



### WRITE OUTPUT FILE
print('Starting writing data...')

# save post descriptives
cleaning_and_mapping_descriptives['rows_post'] = df.shape[0]
cleaning_and_mapping_descriptives['columns_post'] = df.shape[1]
cleaning_and_mapping_descriptives['unique_visitors_post'] = df['visitor_id'].nunique()

df.to_pickle('../data/processed_data/'+output_file, compression='gzip')

print('Writing data complete.')



print('Cleaning and mapping complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time and descriptives dataframe
cleaning_and_mapping_descriptives['run_time'] = run_time.seconds
cleaning_and_mapping_descriptives.to_pickle('../results/descriptives/cleaning_and_mapping_descriptives_'+input_file[:16]+'.pkl.gz', compression='gzip')
