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
print('Starting loading data...')

df = pd.read_pickle('../data/processed_data/'+input_files[0], compression='gzip')
for input_file in input_files[1:]:
    df = df.append(pd.read_pickle('../data/processed_data/'+input_file, compression='gzip'))

print('Loading data complete.')



# aggregate columns
print('Starting aggregating columns...')

# process product items and prices before aggregation
df = process_product_items(df)
df = process_product_item_prices(df)

# sort dataframe by visit_start_time_gmt, visitor_id and visit_num
df = df.sort_values(['visit_start_time_gmt', 'visitor_id', 'visit_num'], ascending=[True, True, True])

# group columns by visitor_id, visit_start_time, visit_num and aggregate
df_aggregated = df.groupby(by = ['visitor_id', 'visit_start_time_gmt', 'visit_num'], as_index=False).agg({'hit_time_gmt': ['min', 'max'],
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
'num_product_items_seen' : 'sum',
'sum_price_product_items_seen' : 'sum',
'standard_search_results_clicked' : 'sum',
'standard_search_started' : 'sum',
'suggested_search_results_clicked' : 'sum',
'country' : ['first', 'last'], 
'cookies' : ['first', 'last'], 
'persistent_cookie' : ['first', 'last'], 
'search_page_num' : ['first', 'last'],
'connection_type' : ['first', 'last'], 
'browser' : ['first', 'last'], 
'operating_system' : ['first', 'last'], 
'search_engine' : ['first', 'last'],
'marketing_channel' : ['first', 'last'], 
'referrer_type' : ['first', 'last'], 
'new_visit' : ['first', 'last'], 
'hourly_visitor' : ['first', 'last'], 
'daily_visitor' : ['first', 'last'], 
'weekly_visitor' : ['first', 'last'], 
'monthly_visitor' : ['first', 'last'], 
'quarterly_visitor' : ['first', 'last'], 
'yearly_visitor' : ['first', 'last'], 
'product_categories' : ['first', 'last'], 
'device_type_user_agent' : ['first', 'last'], 
'device_brand_name_user_agent' : ['first', 'last'], 
'device_operating_system_user_agent' : ['first', 'last'], 
'device_browser_user_agent' : ['first', 'last'],
'repeat_orders' : ['first', 'last'], 
'net_promoter_score' : ['first', 'last'], 
'hit_of_logged_in_user' : ['first', 'last'],
'registered_user' : ['first', 'last'],
'user_gender' : ['first', 'last'], 
'user_age' : ['first', 'last'], 
'visit_during_tv_spot' : ['first', 'last']})

# rename columns
df_aggregated.columns = ['_'.join(x) for x in df_aggregated.columns.ravel()]
df_aggregated.rename(columns={'visitor_id_' : 'visitor_id'}, inplace=True)
df_aggregated.rename(columns={'visitor_id_count' : 'hit_count'}, inplace=True)
df_aggregated.rename(columns={'visit_start_time_gmt_' : 'visit_start_time_gmt'}, inplace=True)
df_aggregated.rename(columns={'visit_num_' : 'visit_num'}, inplace=True)

# calculate hit_count per visit and join with df_aggregated
hit_count = df[['visitor_id', 'visit_start_time_gmt', 'visit_num']].groupby(['visitor_id', 'visit_start_time_gmt', 'visit_num']).size().reset_index(name='hit_count')
df_aggregated = pd.merge(df_aggregated, hit_count, how='left', on=['visitor_id', 'visit_start_time_gmt', 'visit_num'])

print('Aggregating columns complete.')



# write data
print('Starting writing data...')

df_aggregated.to_pickle('../data/processed_data/'+output_file, compression='gzip')

print('Writing data complete.')



# save run time
print('Aggregation complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'aggregation_run_time.txt'
run_time_dict = {'aggregation run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)