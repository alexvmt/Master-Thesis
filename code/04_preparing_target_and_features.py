#!/usr/bin/env python
# coding: utf-8

##### PREPARING TARGET AND FEATURES #####

print('Starting preparing target and features...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd
import re

from helper_functions import *



### LOAD DATA
print('Starting loading data...')

# input file
input_file = 'clickstream_0516-1016_aggregated.pkl.gz'

# output file
output_file = 'clickstream_0516-1016_prepared.pkl.gz'

# load data
df = pd.read_pickle('../data/processed_data/'+input_file)

# set up dataframe for descriptives
columns = ['rows_pre',
'rows_post',
'columns_pre',
'columns_post',
'unique_visitors_pre',
'unique_visitors_post',
'run_time']
index = [input_file[:16]]
preparing_target_and_features_descriptives = pd.DataFrame(index=index, columns=columns)

# save pre descriptives
preparing_target_and_features_descriptives['rows_pre'] = df.shape[0]
preparing_target_and_features_descriptives['columns_pre'] = df.shape[1]
preparing_target_and_features_descriptives['unique_visitors_pre'] = df['visitor_id'].nunique()

print('Loading data complete.')



### PREPARE DATA
print('Starting preparing data...')

# keep only visits with more than 1 hit (drop bounce visits)
df = df[df['visit_page_num_max'] > 1]

# sort dataframe by visitor id, visit num and visit_start time_gmt
df = df.sort_values(['visitor_id', 'visit_num', 'visit_start_time_gmt'], ascending=[True, True, True])

print('Preparing data complete.')



### PREPARE TARGET
print('Starting preparing target...')

# purchase within current visit
df['purchase_within_current_visit'] = df['purchase_boolean_sum'].apply(lambda x: 1 if x >= 1 else 0)

# purchase within next 24 hours
purchases = df[df['purchase_within_current_visit'] == 1][['visitor_id', 'visit_start_time_gmt']]
purchases.rename(columns={'visit_start_time_gmt' : 'purchase_time'}, inplace=True)
visits = df[['visitor_id', 'visit_start_time_gmt']].copy()
purchases_visits = pd.merge(visits, purchases, how='left', on='visitor_id')
purchases_visits = purchases_visits[pd.notnull(purchases_visits['purchase_time'])]

purchases_visits['purchase_time_minus_visit_time'] = purchases_visits['purchase_time'] - purchases_visits['visit_start_time_gmt']
purchases_visits['purchase_time_minus_visit_time_delta_hours'] = purchases_visits['purchase_time_minus_visit_time'].apply(lambda x: x.total_seconds() // 3600)
purchases_visits['purchase_within_next_24_hours'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if (x >= 0) & (x <= 24) else 0)

purchase_within_next_24_hours = purchases_visits[purchases_visits['purchase_within_next_24_hours'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours']]
purchase_within_next_24_hours.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], inplace=True)
df = pd.merge(df, purchase_within_next_24_hours, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_within_next_24_hours'].fillna(0, inplace=True)
df['purchase_within_next_24_hours'] = df['purchase_within_next_24_hours'].astype(np.int64)
 
print('Preparing target complete')



### PREPARE CATEGORICAL FEATURES
print('Starting preparing categorical features...')

# clean categorical features and reduce number levels if necessary/possible
# rule: if categorical feature has 10 or more levels, group levels with less than 0,1% of frequency compared to most frequent level in 'Other' level
df = process_product_categories(df)
df = process_net_promoter_score(df)
df = process_user_gender(df)
df = process_user_age(df)
df = process_search_engines(df)
df = process_device_types(df)
df = process_device_brand_names(df)
df = process_device_operating_systems(df)
df = process_device_browsers(df)

# one hot encode categorical features
categorical_features = ['connection_type_first',
'marketing_channel_first',
'referrer_type_first',
'net_promoter_score_first',
'user_gender_first',
'product_categories_first_level_1',
'search_engine_first_reduced',
'device_type_user_agent_first_reduced',
'device_brand_name_user_agent_first_reduced',
'device_operating_system_user_agent_first_reduced',
'device_browser_user_agent_first_reduced']
dummies = pd.get_dummies(df.loc[:, df.columns.isin(categorical_features)], drop_first=True)
df.drop(categorical_features, axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

# create bins for user age
df['user_age_14-25_first'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1991) & (x <= 2002) else 0)
df['user_age_26-35_first'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1981) & (x <= 1990) else 0)
df['user_age_36-45_first'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1971) & (x <= 1980) else 0)
df['user_age_46-55_first'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1961) & (x <= 1970) else 0)
df['user_age_56-65_first'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1951) & (x <= 1960) else 0)
df['user_age_65_plus_first'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1900) & (x <= 1950) else 0)

# flag to indicate visit from Switzerland since most visits and purchases are from Switzerland
df['Switzerland_first'] = df['country_first'].apply(lambda x: 1 if x == 'Switzerland' else 0)

print('Preparing categorical features complete')



### CREATE TIME FEATURES
print('Starting creating time features...')

# visit duration in seconds
df['visit_duration_seconds'] = df['hit_time_gmt_max'] - df['hit_time_gmt_min']
df['visit_duration_seconds'] = df['visit_duration_seconds'].apply(lambda x: x.seconds)

# month
df['month'] = df['date_time_min'].apply(lambda x: x.month)
df['may'] = df['month'].apply(lambda x: 1 if x == 5 else 0)
df['june'] = df['month'].apply(lambda x: 1 if x == 6 else 0)
df['july'] = df['month'].apply(lambda x: 1 if x == 7 else 0)
df['august'] = df['month'].apply(lambda x: 1 if x == 8 else 0)
df['september'] = df['month'].apply(lambda x: 1 if x == 9 else 0)
df['october'] = df['month'].apply(lambda x: 1 if x == 10 else 0)

# day of month
df['day_of_month'] = df['date_time_min'].apply(lambda x: x.day)
df['beginning_of_month'] = df['day_of_month'].apply(lambda x: 1 if x <= 10 else 0)
df['middle_of_month'] = df['day_of_month'].apply(lambda x: 1 if (x >= 11) & (x <= 20) else 0)
df['end_of_month'] = df['day_of_month'].apply(lambda x: 1 if (x >= 21) else 0)

# day of week
df['day_of_week'] = df['date_time_min'].apply(lambda x: x.weekday())
df['weekday'] = df['day_of_week'].apply(lambda x: 1 if x <= 4 else 0)
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# hour of day
df['hour_of_day'] = df['date_time_min'].apply(lambda x: x.hour)
df['morning'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 6) & (x <= 11) else 0)
df['afternoon'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 12) & (x <= 17) else 0)
df['evening'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 18) & (x <= 23) else 0)
df['night'] = df['hour_of_day'].apply(lambda x: 1 if (x >= 23) | (x <= 5) else 0)

print('Creating time features complete')



### VISITS, PURCHASES, PAGE VIEWS AND PRODUCT VIEWS IN LAST N HOURS/DAYS
print('Starting preparing visits, purchases, page views and product views in last n hours/days...')

# sort dataframe by visitor id, visit num and visit start time
df = df.sort_values(['visitor_id', 'visit_num', 'visit_start_time_gmt'], ascending=[True, True, True])

# hours and days since last visit
df['visitor_id_lag'] = df['visitor_id'].shift(1)
df['visit_start_time_gmt_lag'] = df['visit_start_time_gmt'].shift(1)
df['visit_start_time_gmt_minus_visit_start_time_gmt_lag'] = df['visit_start_time_gmt'] - df['visit_start_time_gmt_lag']
df['days_since_last_visit'] = df.apply(lambda x: x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'].days if (pd.notnull(x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'])) & (x['visitor_id'] == x['visitor_id_lag']) else np.nan, axis=1)
df['hours_since_last_visit'] = df.apply(lambda x: x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'].seconds // 3600 if (pd.notnull(x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'])) & (x['visitor_id'] == x['visitor_id_lag']) else np.nan, axis=1)
df['visit_in_last_24_hours'] = df['hours_since_last_visit'].apply(lambda x: 1 if (x >= 1) & (x <= 24) else 0)
df['visit_in_last_7_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x >= 1) & (x <= 7) else 0)
df['visit_in_last_30_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x >= 8) & (x <= 30) else 0)
df['visit_in_last_30_plus_days'] = df['days_since_last_visit'].apply(lambda x: 1 if x >= 31 else 0)

# page views in last visit
df['page_view_boolean_sum_lag'] = df['page_view_boolean_sum'].shift(1)
df['page_views_in_last_visit'] = df.apply(lambda x: x['page_view_boolean_sum_lag'] if x['visitor_id'] == x['visitor_id_lag'] else np.nan, axis=1)
df['page_views_in_last_visit_0'] = df['page_views_in_last_visit'].apply(lambda x: 1 if x == 0 else 0)
df['page_views_in_last_visit_1'] = df['page_views_in_last_visit'].apply(lambda x: 1 if x == 1 else 0)
df['page_views_in_last_visit_2-5'] = df['page_views_in_last_visit'].apply(lambda x: 1 if (x >= 2) & (x <= 5) else 0)
df['page_views_in_last_visit_6-10'] = df['page_views_in_last_visit'].apply(lambda x: 1 if (x >= 6) & (x <= 10) else 0)
df['page_views_in_last_visit_11-20'] = df['page_views_in_last_visit'].apply(lambda x: 1 if (x >= 11) & (x <= 20) else 0)
df['page_views_in_last_visit_20_plus'] = df['page_views_in_last_visit'].apply(lambda x: 1 if x >= 21 else 0)

# product views in last visit
df['product_view_boolean_sum_lag'] = df['product_view_boolean_sum'].shift(1)
df['product_views_in_last_visit'] = df.apply(lambda x: x['product_view_boolean_sum_lag'] if x['visitor_id'] == x['visitor_id_lag'] else np.nan, axis=1)
df['product_views_in_last_visit_0'] = df['product_views_in_last_visit'].apply(lambda x: 1 if x == 0 else 0)
df['product_views_in_last_visit_1'] = df['product_views_in_last_visit'].apply(lambda x: 1 if x == 1 else 0)
df['product_views_in_last_visit_2-5'] = df['product_views_in_last_visit'].apply(lambda x: 1 if (x >= 2) & (x <= 5) else 0)
df['product_views_in_last_visit_6-10'] = df['product_views_in_last_visit'].apply(lambda x: 1 if (x >= 6) & (x <= 10) else 0)
df['product_views_in_last_visit_11-20'] = df['product_views_in_last_visit'].apply(lambda x: 1 if (x >= 11) & (x <= 20) else 0)
df['product_views_in_last_visit_20_plus'] = df['product_views_in_last_visit'].apply(lambda x: 1 if x >= 21 else 0)

# hours and days since last purchase
purchases_visits['purchase_in_last_24_hours'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if (x <= -1) & (x >= -24) else 0)
purchases_visits['purchase_in_last_7_days'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if (x <= -25) & (x >= -168) else 0)
purchases_visits['purchase_in_last_30_days'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if (x <= -168) & (x >= -720) else 0)
purchases_visits['purchase_in_last_30_plus_days'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if x <= -721 else 0)

purchases_visits.rename(columns={'visit_time' : 'visit_start_time_gmt'}, inplace=True)
purchase_in_last_24_hours = purchases_visits[purchases_visits['purchase_in_last_24_hours'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_24_hours']]
purchase_in_last_24_hours.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], inplace=True)
df = pd.merge(df, purchase_in_last_24_hours, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_24_hours'].fillna(0, inplace=True)
df['purchase_in_last_24_hours'] = df['purchase_in_last_24_hours'].astype(np.int64)

purchase_in_last_7_days = purchases_visits[purchases_visits['purchase_in_last_7_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_7_days']]
purchase_in_last_7_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], inplace=True)
df = pd.merge(df, purchase_in_last_7_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_7_days'].fillna(0, inplace=True)
df['purchase_in_last_7_days'] = df['purchase_in_last_7_days'].astype(np.int64)

purchase_in_last_30_days = purchases_visits[purchases_visits['purchase_in_last_30_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_30_days']]
purchase_in_last_30_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], inplace=True)
df = pd.merge(df, purchase_in_last_30_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_30_days'].fillna(0, inplace=True)
df['purchase_in_last_30_days'] = df['purchase_in_last_30_days'].astype(np.int64)

purchase_in_last_30_plus_days = purchases_visits[purchases_visits['purchase_in_last_30_plus_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_30_plus_days']]
purchase_in_last_30_plus_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], inplace=True)
df = pd.merge(df, purchase_in_last_30_plus_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_30_plus_days'].fillna(0, inplace=True)
df['purchase_in_last_30_plus_days'] = df['purchase_in_last_30_plus_days'].astype(np.int64)

print('Preparing visits, purchases, page views and product views in last n hours/days complete')



### WRITE DATA
print('Starting writing data...')

# drop columns not needed for modeling
columns_to_drop = ['purchase_boolean_sum',
'purchase_within_current_visit',
'checkout_boolean_sum',
'month',
'day_of_month',
'day_of_week',
'hour_of_day',
'hit_time_gmt_min',
'hit_time_gmt_max',
'date_time_min',
'date_time_max',
'user_age_first', 
'country_first', 
'visitor_id_lag', 
'visit_start_time_gmt_lag', 
'visit_start_time_gmt_minus_visit_start_time_gmt_lag', 
'days_since_last_visit', 
'hours_since_last_visit',
'page_view_boolean_sum_lag',
'page_views_in_last_visit',
'product_view_boolean_sum_lag',
'product_views_in_last_visit']
df.drop(columns_to_drop, axis=1, inplace=True)

# save post descriptives
preparing_target_and_features_descriptives['rows_post'] = df.shape[0]
preparing_target_and_features_descriptives['columns_post'] = df.shape[1]
preparing_target_and_features_descriptives['unique_visitors_post'] = df['visitor_id'].nunique()

df.to_pickle('../data/processed_data/'+output_file, compression='gzip')

print('Writing data complete.')



print('Preparing target and features complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time and descriptives dataframe
preparing_target_and_features_descriptives['run_time'] = run_time.seconds
preparing_target_and_features_descriptives.to_pickle('../results/descriptives/preparing_target_and_features_descriptives.pkl.gz', compression='gzip')
