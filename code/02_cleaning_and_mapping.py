#!/usr/bin/env python
# coding: utf-8

##### CLEANING AND MAPPING #####

print('Starting cleaning and mapping...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd

from helper_functions import *

# input files
input_files = ['clickstream_0516_raw.tsv.gz',
               'clickstream_0616_raw.tsv.gz',
               'clickstream_0716_raw.tsv.gz',
               'clickstream_0816_raw.tsv.gz',
               'clickstream_0916_raw.tsv.gz',
               'clickstream_1016_raw.tsv.gz']

# output files
output_files = ['clickstream_0516_cleaned_and_mapped.pkl.gz',
                'clickstream_0616_cleaned_and_mapped.pkl.gz',
                'clickstream_0716_cleaned_and_mapped.pkl.gz',
                'clickstream_0816_cleaned_and_mapped.pkl.gz',
                'clickstream_0916_cleaned_and_mapped.pkl.gz',
                'clickstream_1016_cleaned_and_mapped.pkl.gz']

# empty lists to save number of raw and cleaned hits
number_raw_hits = []
number_cleaned_hits = []

# process input files and write output files
for input_file, output_file in zip(input_files, output_files):

    print('Processing '+input_file+'...')
    print('Starting loading data...')

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

    # append number of raw hits to raw hits list
    number_raw_hits.append(df.shape[0])

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

    # append number of cleaned hits to cleaned hits list
    number_cleaned_hits.append(df.shape[0])

    # write data
    print('Starting writing data...')

    df.to_pickle('../data/processed_data/'+output_file, compression='gzip')

    print('Writing data complete.')

    print('Processing '+input_file+' complete.')



# save number of raw hits
save_descriptives('../results/descriptives/number_raw_hits.txt', number_raw_hits)

# save number of cleaned hits
save_descriptives('../results/descriptives/number_cleaned_hits.txt', number_cleaned_hits)



print('Cleaning and mapping complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time
save_descriptives('../results/descriptives/cleaning_and_mapping_run_time.txt', run_time)