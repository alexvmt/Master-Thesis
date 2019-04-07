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
output_file = params[2]

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

print('Loading data complete.')



### PROCESS INPUTE FILE
print('Processing '+input_file+'...')

# set up dataframe for number of raw and cleaned hits
columns = ['raw_hits',
'cleaned_hits',
'absolute_difference',
'relative_difference']
index = [input_file[:16]]
raw_and_cleaned_hits = pd.DataFrame(index=index, columns=columns)

# save number of raw hits
raw_and_cleaned_hits['raw_hits'] = df.shape[0]

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

# save number of cleaned hits and calculate differences
raw_and_cleaned_hits['cleaned_hits'] = df.shape[0]
raw_and_cleaned_hits['absolute_difference'] = raw_and_cleaned_hits['raw_hits'] - raw_and_cleaned_hits['cleaned_hits']
raw_and_cleaned_hits['relative_difference'] = raw_and_cleaned_hits['absolute_difference'] / raw_and_cleaned_hits['raw_hits']
raw_and_cleaned_hits['relative_difference'] = raw_and_cleaned_hits['relative_difference'].apply(lambda x: round(x, 4))

# save raw and cleaned hits dataframe
raw_and_cleaned_hits.to_pickle('../results/descriptives/raw_and_cleaned_hits_'+input_file[:16]+'.pkl.gz', compression='gzip')

print('Processing '+input_file+' complete.')



### WRITE OUTPUT FILE
print('Starting writing data...')

df.to_pickle('../data/processed_data/'+output_file, compression='gzip')

print('Writing data complete.')



print('Cleaning and mapping complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/cleaning_and_mapping_run_time_'+input_file[:16]+'.txt', run_time)