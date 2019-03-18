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



# input file
input_file = 'clickstream_0516-1016_cleaned_and_mapped.tsv.gz'

# output file
output_file = 'clickstream_0516-1016_aggregated.tsv.gz'

# load data
print('Starting loading data...')

data_types = {'visitor_id': 'str', 
'visit_num': 'int', 
'visit_page_num': 'int', 
'purchase_boolean': 'int',
'product_view_boolean': 'int', 
'checkout_boolean': 'int', 
'cart_addition_boolean': 'int',
'cart_removal_boolean': 'int', 
'cart_view_boolean': 'int', 
'campaign_view_boolean': 'int',
'cart_value': 'float', 
'page_view_boolean': 'int', 
'last_purchase_num': 'int',
'hit_counter': 'int', 
'standard_search_results_clicked': 'int', 
'standard_search_started': 'int',
'suggested_search_results_clicked': 'int', 
'country': 'str', 
'cookies': 'int',
'persistent_cookie': 'int', 
'search_page_num': 'int',
'connection_type': 'str', 
'browser': 'str',
'operating_system': 'str', 
'search_engine': 'str',
'marketing_channel': 'str', 
'referrer_type': 'str',
'new_visit': 'int', 
'hourly_visitor': 'int',
'daily_visitor': 'int', 
'weekly_visitor': 'int',
'monthly_visitor': 'int', 
'quarterly_visitor': 'int',
'yearly_visitor': 'int', 
'product_items': 'str',
'product_item_price': 'str', 
'product_categories': 'str',
'device_type_user_agent': 'str', 
'device_brand_name_user_agent': 'str',
'device_operating_system_user_agent': 'str', 
'device_browser_user_agent': 'str',
'repeat_orders': 'int',
'net_promoter_score': 'str', 
'hit_of_logged_in_user': 'int',
'registered_user': 'int',
'user_gender': 'str', 
'user_age': 'str',
'visit_during_tv_spot': 'int'}

date_columns = ['date_time', 
'hit_time_gmt', 
'visit_start_time_gmt']

df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, dtype=data_types, parse_dates=date_columns)

print('Loading data complete.')



# aggregate numerical columns
print('Starting aggregating numerical columns...')

# process product items and prices before aggregation
df = process_product_items(df)
df = process_product_item_prices(df)

# select numerical columns
numerical_columns_names = ['visitor_id', 
'visit_start_time_gmt',
'hit_time_gmt',
'date_time',
'visit_num', 
'visit_page_num',
'purchase_boolean', 
'product_view_boolean', 
'checkout_boolean', 
'cart_addition_boolean', 
'cart_removal_boolean', 
'cart_view_boolean', 
'campaign_view_boolean', 
'cart_value', 
'page_view_boolean', 
'last_purchase_num', 
'num_product_items_seen', 
'sum_price_product_items_seen', 
'hit_counter', 
'standard_search_results_clicked', 
'standard_search_started', 
'suggested_search_results_clicked']
numerical_columns = df.loc[:, df.columns.isin(numerical_columns_names)].copy()

# group numerical columns by visitor_id, visit_num and visit_start_time and aggregate
numerical_columns_aggregated = numerical_columns.groupby(by = ['visitor_id', 'visit_num', 'visit_start_time_gmt'], as_index=False).agg({'hit_time_gmt': ['min', 'max'],
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
'hit_counter' : 'sum',
'standard_search_results_clicked' : 'sum',
'standard_search_started' : 'sum',
'suggested_search_results_clicked' : 'sum'})

# rename columns
numerical_columns_aggregated.columns = ['_'.join(x) for x in numerical_columns_aggregated.columns.ravel()]
numerical_columns_aggregated = numerical_columns_aggregated.rename(columns={'visitor_id_' : 'visitor_id', 
'visit_start_time_gmt_' : 'visit_start_time_gmt',
'hit_time_gmt_min' : 'hit_time_gmt',
'hit_time_gmt_max' : 'last_hit_time_gmt_visit',
'date_time_min' : 'date_time',
'date_time_max' : 'last_date_time_visit',
'visit_num_' : 'visit_num',
'visit_page_num_max' : 'visit_page_num',
'purchase_boolean_sum' : 'purchase',
'product_view_boolean_sum' : 'product_views',
'checkout_boolean_sum' : 'checkouts',
'cart_addition_boolean_sum' : 'cart_additions',
'cart_removal_boolean_sum' : 'cart_removals',
'cart_view_boolean_sum' : 'cart_views',
'campaign_view_boolean_sum' : 'campaign_views',
'cart_value_sum' : 'cart_value',
'page_view_boolean_sum' : 'page_views',
'last_purchase_num_max' : 'last_purchase_num',
'num_product_items_seen_sum' : 'num_product_items_seen',
'sum_price_product_items_seen_sum' : 'sum_price_product_items_seen',
'hit_counter_sum' : 'hit_count',
'standard_search_results_clicked_sum' : 'standard_search_results_clicked', 
'standard_search_started_sum' : 'standard_search_started', 
'suggested_search_results_clicked_sum' : 'suggested_search_results_clicked'})

# sort by visit_start_time_gmt, visitor_id and visit_num
numerical_columns_aggregated = numerical_columns_aggregated.sort_values(['visit_start_time_gmt', 'visitor_id', 'visit_num'], ascending=[True, True, True])

print('Aggregating numerical columns complete.')



# prepare categorical columns for merging
print('Starting preparing categorical columns...')

# select categorical columns
categorical_columns_names = ['visitor_id',
'visit_num',
'visit_start_time_gmt', 
'country', 
'cookies', 
'persistent_cookie', 
'search_page_num',
'connection_type', 
'browser', 
'operating_system', 
'search_engine',
'marketing_channel', 
'referrer_type', 
'new_visit', 
'hourly_visitor', 
'daily_visitor', 
'weekly_visitor', 
'monthly_visitor', 
'quarterly_visitor', 
'yearly_visitor', 
'product_categories', 
'device_type_user_agent', 
'device_brand_name_user_agent', 
'device_operating_system_user_agent', 
'device_browser_user_agent',
'repeat_orders', 
'net_promoter_score', 
'hit_of_logged_in_user',
'registered_user',
'user_gender', 
'user_age', 
'visit_during_tv_spot']
categorical_columns = df.loc[:, df.columns.isin(categorical_columns_names)].copy()

# sort by visit_start_time_gmt, visitor_id and visit_num
categorical_columns = categorical_columns.sort_values(['visit_start_time_gmt', 'visitor_id', 'visit_num'], ascending=[True, True, True])

print('Preparing categorical columns complete.')



# merging numerical and categorical columns to visit level dataframe
print('Starting merging numerical and categorical columns...')

df = pd.merge_asof(numerical_columns_aggregated, categorical_columns, on='visit_start_time_gmt', by='visitor_id')

print('Merging numerical and categorical columns complete.')



# write data
print('Starting writing data...')

df.to_csv('../data/processed_data/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)

print('Writing data complete.')



# save run time
print('Aggregation complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'aggregation_run_time.txt'
run_time_dict = {'aggregation run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)