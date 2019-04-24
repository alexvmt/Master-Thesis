#!/usr/bin/env python
# coding: utf-8

##### AGGREGATION #####

print('Starting aggregation...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd

from helper_functions import *



### LOAD DATA
print('Starting loading data...')

# input files
input_files = ['clickstream_0516_cleaned_and_mapped.pkl.gz',
'clickstream_0616_cleaned_and_mapped.pkl.gz',
'clickstream_0716_cleaned_and_mapped.pkl.gz',
'clickstream_0816_cleaned_and_mapped.pkl.gz',
'clickstream_0916_cleaned_and_mapped.pkl.gz',
'clickstream_1016_cleaned_and_mapped.pkl.gz']

# output file
output_file = 'clickstream_0516-1016_aggregated.pkl.gz'

# load data
df = pd.read_pickle('../data/processed_data/'+input_files[0])
for input_file in input_files[1:]:
    df = df.append(pd.read_pickle('../data/processed_data/'+input_file))

# set up dataframe for descriptives
columns = ['rows_pre',
'rows_post',
'columns_pre',
'columns_post',
'unique_visitors_pre',
'unique_visitors_post',
'run_time']
index = [input_file[:16]]
aggregation_descriptives = pd.DataFrame(index=index, columns=columns)

# save pre descriptives
aggregation_descriptives['rows_pre'] = df.shape[0]
aggregation_descriptives['columns_pre'] = df.shape[1]
aggregation_descriptives['unique_visitors_pre'] = df['visitor_id'].nunique()

print('Loading data complete.')



### AGGREGATE COLUMNS
print('Starting aggregating columns...')

# process product items and prices before aggregation
df = process_product_items(df)
df = process_product_item_prices(df)

# sort dataframe by visit start time, visitor id and visit num
df = df.sort_values(['visit_start_time_gmt', 'visitor_id', 'visit_num'], ascending=[True, True, True])

# group columns by visitor id, visit start time, visit num and aggregate
df = df.groupby(by = ['visitor_id', 'visit_start_time_gmt', 'visit_num'], as_index=False).agg({'hit_time_gmt': ['min', 'max'],
'date_time' : ['min', 'max'],
'visit_page_num' : 'max',
'purchase_boolean' : 'sum',
'product_view_boolean' : 'sum',
'checkout_boolean' : 'sum',
'cart_addition_boolean': 'sum',
'cart_removal_boolean': 'sum',
'cart_view_boolean': 'sum',
'campaign_view_boolean': 'sum',
'cart_value': 'sum',
'page_view_boolean': 'sum',
'last_purchase_num': 'max',
'product_items' : 'sum',
'product_item_price' : 'sum',
'standard_search_results_clicked' : 'sum',
'standard_search_started' : 'sum',
'suggested_search_results_clicked' : 'sum',
'country' : 'first', 
'cookies' : 'first', 
'persistent_cookie' : 'first', 
'search_page_num' : 'first',
'connection_type' : 'first', 
'search_engine' : 'first',
'marketing_channel' : 'first', 
'referrer_type' : 'first', 
'new_visit' : 'first', 
'hourly_visitor' : 'first', 
'daily_visitor' : 'first', 
'weekly_visitor' : 'first', 
'monthly_visitor' : 'first', 
'quarterly_visitor' : 'first', 
'yearly_visitor' : 'first', 
'product_categories' : 'first', 
'device_type_user_agent' : 'first', 
'device_brand_name_user_agent' : 'first', 
'device_operating_system_user_agent' : 'first', 
'device_browser_user_agent' : 'first',
'repeat_orders' : 'first', 
'net_promoter_score' : 'first', 
'hit_of_logged_in_user' : 'first',
'registered_user' : 'first',
'user_gender' : 'first', 
'user_age' : 'first', 
'visit_during_tv_spot' : 'first'})

# rename columns
df.columns = ['_'.join(x) for x in df.columns.ravel()]
df.rename(columns={'visitor_id_' : 'visitor_id'}, inplace=True)
df.rename(columns={'visitor_id_count' : 'hit_count'}, inplace=True)
df.rename(columns={'visit_start_time_gmt_' : 'visit_start_time_gmt'}, inplace=True)
df.rename(columns={'visit_num_' : 'visit_num'}, inplace=True)

print('Aggregating columns complete.')



### WRITE DATA
print('Starting writing data...')

# save post descriptives
aggregation_descriptives['rows_post'] = df.shape[0]
aggregation_descriptives['columns_post'] = df.shape[1]
aggregation_descriptives['unique_visitors_post'] = df['visitor_id'].nunique()

df.to_pickle('../data/processed_data/'+output_file, compression='gzip')

print('Writing data complete.')



print('Aggregation complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time and descriptives dataframe
aggregation_descriptives['run_time'] = run_time.seconds
aggregation_descriptives.to_pickle('../results/descriptives/aggregation_descriptives.pkl.gz', compression='gzip')
