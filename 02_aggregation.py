#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### AGGREGATION #####


# In[2]:


print('Starting aggregation...')


# In[3]:


### import libraries
import pandas as pd
import numpy as np
from datetime import datetime,date

start_time = datetime.now()
print('Start time: ', start_time)


# In[4]:


#### SELECT INPUT AND OUTPUT FILES


# In[5]:


input_file = '3_day_sample_cleaned_and_mapped.tsv.gz'
output_file = '3_day_sample_aggregated.tsv.gz'
#input_data = '6_week_sample_cleaned_and_mapped.tsv.gz'
#output_data = '6_week_sample_aggregated.tsv.gz'
#input_data = '12_week_sample_cleaned_and_mapped.tsv.gz'
#output_data = '12_week_sample_aggregated.tsv.gz'
#input_data = '25_week_sample_cleaned_and_mapped.tsv.gz'
#output_data = '25_week_sample_aggregated.tsv.gz'

print('Input file selected: ', input_file)
print('Output file selected', output_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, parse_dates=['hit_time_gmt'])

print('Loading data complete.')


# In[8]:


##### CLEAN DATA
print('Cleaning data...')


# In[9]:


##### AGGREGATE NUMERICAL COLUMNS
print('Aggregating numerical columns...')


# In[10]:


# select numerical columns
numerical_cols_names = ['visitor_id', 
                        'visit_num', 
                        'visit_page_num', 
                        'hit_time_gmt',
                        'purchase_boolean', 
                        'product_view_boolean', 
                        'checkout_boolean', 
                        'cart_addition_boolean', 
                        'cart_removal_boolean', 
                        'cart_view_boolean', 
                        'campaign_view_boolean', 
                        'cart_value_(v50)', 
                        'page_view_boolean', 
                        'last_purchase_num', 
                        'num_product_items_seen', 
                        'sum_price_product_items_seen', 
                        'server_call_counter_(e1)', 
                        'int._stand._search_result_clicked_(e16)', 
                        'active_stand._search_started_(e17)', 
                        'sugg._search_result_clicked_(e18)']

numerical_cols = df.loc[:, df.columns.isin(numerical_cols_names)].copy()

# group numerical columns by visitor_id and visit_num and aggregate
numerical_cols_aggregated = numerical_cols.groupby(by = ['visitor_id', 'visit_num'], as_index=False).agg({'visit_page_num' : 'max',
                                                                                                          'hit_time_gmt': ['min', 'max'],
                                                                                                          'purchase_boolean' : 'sum',
                                                                                                          'product_view_boolean' : 'sum',
                                                                                                          'checkout_boolean' : 'sum',
                                                                                                          'cart_addition_boolean': 'sum',
                                                                                                          'cart_removal_boolean': 'sum',
                                                                                                          'cart_view_boolean': 'sum',
                                                                                                          'campaign_view_boolean': 'sum',
                                                                                                          'cart_value_(v50)': 'sum',
                                                                                                          'page_view_boolean': 'sum',
                                                                                                          'last_purchase_num': 'max',
                                                                                                          'num_product_items_seen' : 'sum',
                                                                                                          'sum_price_product_items_seen' : 'sum',
                                                                                                          'server_call_counter_(e1)' : 'sum',
                                                                                                          'int._stand._search_result_clicked_(e16)' : 'sum',
                                                                                                          'active_stand._search_started_(e17)' : 'sum',
                                                                                                          'sugg._search_result_clicked_(e18)' : 'sum'})

# rename columns
numerical_cols_aggregated.columns = ['_'.join(x) for x in numerical_cols_aggregated.columns.ravel()]
numerical_cols_aggregated = numerical_cols_aggregated.rename(columns={'visitor_id_' : 'visitor_id',
                                                                      'visit_num_' : 'visit_num', 
                                                                      'visit_page_num_max' : 'visit_page_num',
                                                                      'hit_time_gmt_min' : 'hit_time_gmt',
                                                                      'hit_time_gmt_max' : 'last_hit_time_gmt_visit',
                                                                      'purchase_boolean_sum' : 'purchases',
                                                                      'product_view_boolean_sum' : 'product_views',
                                                                      'checkout_boolean_sum' : 'checkouts',
                                                                      'cart_addition_boolean_sum' : 'cart_additions',
                                                                      'cart_removal_boolean_sum' : 'cart_removals',
                                                                      'cart_view_boolean_sum' : 'cart_views',
                                                                      'campaign_view_boolean_sum' : 'campaign_views',
                                                                      'cart_value_(v50)_sum' : 'cart_value',
                                                                      'page_view_boolean_sum' : 'page_views',
                                                                      'last_purchase_num_max' : 'last_purchase_num',
                                                                      'num_product_items_seen_sum' : 'num_product_items_seen',
                                                                      'sum_price_product_items_seen_sum' : 'sum_price_product_items_seen',
                                                                      'server_call_counter_(e1)_sum' : 'hit_count',
                                                                      'int._stand._search_result_clicked_(e16)_sum' : 'standard_search_results_clicked', 
                                                                      'active_stand._search_started_(e17)_sum' : 'standard_searches_started', 
                                                                      'sugg._search_result_clicked_(e18)_sum' : 'suggested_search_results_clicked'})

# sort by hit_time_gmt, last_hit_time_gmt_visit, visitor_id and visit_num
numerical_cols_aggregated = numerical_cols_aggregated.sort_values(['hit_time_gmt', 
                                                                   'last_hit_time_gmt_visit', 
                                                                   'visitor_id', 
                                                                   'visit_num'], ascending=[True, True, True, True])

# reset index to make sure that index values are unique
numerical_cols_aggregated = numerical_cols_aggregated.reset_index(drop=True)

print('Aggregating numerical columns complete.')


# In[11]:


##### PROCESS CATEGORICAL COLUMNS
print('Processing categorical columns...')


# In[12]:


# select categorical columns
categorical_cols_names = ['visitor_id',
                          'hit_time_gmt',  
                          'country', 
                          'geo_region', 
                          'geo_city', 
                          'geo_zip', 
                          'geo_dma',
                          'post_channel', 
                          'post_cookies', 
                          'post_persistent_cookie', 
                          'search_page_num',
                          'connection_type', 
                          'browser', 
                          'operating_system_generalized', 
                          'search_engine_generalized', 
                          'marketing_channel', 
                          'referrer_type', 
                          'repeat_orders_(e9)', 
                          'net_promoter_score_raw_(v10)_-_user', 
                          'registration_(any_form)_(e20)', 
                          'hit_of_logged_in_user_(e23)', 
                          'newsletter_signup_(any_form)_(e26)', 
                          'newsletter_subscriber_(e27)', 
                          'registered_user', 
                          'login_status', 
                          'user_gender_(v61)', 
                          'user_age_(v62)', 
                          'visit_during_tv_spot_(e71)', 
                          'login_success_(e72)', 
                          'logout_success_(e73)', 
                          'login_fail_(e74)', 
                          'registration_fail_(e75)',
                          'new_visit', 
                          'hourly_visitor', 
                          'daily_visitor', 
                          'weekly_visitor', 
                          'monthly_visitor', 
                          'quarterly_visitor', 
                          'yearly_visitor', 
                          'product_categories_level_1', 
                          'product_categories_level_2', 
                          'product_categories_level_3']

categorical_cols = df.loc[:, df.columns.isin(categorical_cols_names)].copy()

# sort by hit_time_gmt, visitor_id and visit_num
categorical_cols = categorical_cols.sort_values(['hit_time_gmt', 
                                                 'visitor_id'], ascending=[True, True])

# reset index to make sure that index values are unique
categorical_cols = categorical_cols.reset_index(drop=True)

print('Preparing categorical columns complete.')


# In[13]:


##### MERGE NUMERICAL AND CATEGORICAL COLUMNS
print('Merging numerical and categorical columns...')


# In[14]:


df = pd.merge_asof(numerical_cols_aggregated, categorical_cols, on='hit_time_gmt', by='visitor_id')

# reset index to make sure that index values are unique
df = df.reset_index(drop=True)

print('Merging numerical and categorical columns complete.')


# In[15]:


##### WRITE DATAFRAME TO FILE
print('Writing dataframe to file...')


# In[16]:


df.to_csv('../data/processed_data/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[17]:


print('Aggregation complete.')
print('Run time: ', datetime.now() - start_time)

