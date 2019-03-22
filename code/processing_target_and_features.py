#!/usr/bin/env python
# coding: utf-8

##### PROCESSING TARGET AND FEATURES #####

print('Starting processing target and features...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd
import re

from helper_functions import *



# input file
input_file = 'clickstream_0516-1016_aggregated.pkl.gz'

# output file
output_file = 'clickstream_0516-1016_processed_final.pkl.gz'

# load data
print('Starting loading data...')

df = pd.read_pickle('../data/processed_data/'+input_file, compression='gzip')

print('Loading data complete.')



### PREPARE DATA
# keep only visits with more than 1 hit
df = df[df['hit_count'] > 1]

# sort dataframe by visitor_id, visit_num and visit_start_time_gmt
df = df.sort_values(['visitor_id', 'visit_num', 'visit_start_time_gmt'], ascending=[True, True, True])



### PROCESS TARGET
print('Starting processing target...')

# purchase within current visit
df['purchase_within_current_visit'] = df['purchase_boolean_sum'].apply(lambda x: 1 if x >= 1 else 0)

# purchase within next visit
df['purchase_within_next_visit'] = df['purchase_within_current_visit']
df['visitor_id_lead'] = df['visitor_id'].shift(-1)
df['purchase_within_current_visit_lead'] = df['purchase_within_current_visit'].shift(-1)
df['purchase_within_next_visit'] = df.apply(lambda x: 1 if (x['visitor_id'] == x['visitor_id_lead']) & (x['purchase_within_current_visit_lead'] == 1) else x['purchase_within_next_visit'], axis=1)

# purchase within next 24 hours and purchase within next 7 days
purchases = df[df['purchase_within_current_visit'] == 1][['visitor_id', 'visit_start_time_gmt']]
purchases.rename(columns={'visit_start_time_gmt' : 'purchase_time'}, inplace=True)
visits = df[['visitor_id', 'visit_start_time_gmt']].copy()
visits.rename(columns={'visit_start_time_gmt' : 'visit_time'}, inplace=True)
purchases_visits = pd.merge(visits, purchases, how='left', on='visitor_id')
purchases_visits = purchases_visits[pd.notnull(purchases_visits['purchase_time'])]

purchases_visits['purchase_time_minus_visit_time'] = purchases_visits['purchase_time'] - purchases_visits['visit_time']
purchases_visits['purchase_time_minus_visit_time_delta_hours'] = purchases_visits['purchase_time_minus_visit_time'].apply(lambda x: x.total_seconds() // 3600)
purchases_visits['purchase_within_next_24_hours'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if (x <= 24) & (x >= 0) else 0)
purchases_visits['purchase_within_next_7_days'] = purchases_visits['purchase_time_minus_visit_time_delta_hours'].apply(lambda x: 1 if (x <= 168) & (x >= 0) else 0)

purchases_visits.rename(columns={'visit_time' : 'visit_start_time_gmt'}, inplace=True)
purchase_within_next_24_hours = purchases_visits[purchases_visits['purchase_within_next_24_hours'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours']]
purchase_within_next_24_hours.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_within_next_24_hours, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_within_next_24_hours'].fillna(0, inplace=True)
df['purchase_within_next_24_hours'] = df['purchase_within_next_24_hours'].astype(np.int64)
 
purchase_within_next_7_days = purchases_visits[purchases_visits['purchase_within_next_7_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_7_days']]
purchase_within_next_7_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_within_next_7_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_within_next_7_days'].fillna(0, inplace=True)
df['purchase_within_next_7_days'] = df['purchase_within_next_7_days'].astype(np.int64)

print('Processing target complete')



### PROCESS CATEGORICAL FEATURES
print('Starting processing categorical features...')

# clean and reduce levels of categorical columns if necessary
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

# encode other categorical features (correlated and static features, fill gaps between first and last for static features like age or gender)
categorical_features = ['country_first',
'country_last',
'connection_type_first',
'connection_type_last',
'marketing_channel_first',
'marketing_channel_last',
'referrer_type_first',
'referrer_type_last',
'net_promoter_score_first',
'net_promoter_score_last',
'user_gender_first',
'user_gender_last',
'product_categories_level_1_first',
'product_categories_level_1_last',
'search_engine_reduced_first',
'search_engine_reduced_last',
'device_type_user_agent_reduced_first',
'device_type_user_agent_reduced_last',
'device_brand_name_user_agent_reduced_first',
'device_brand_name_user_agent_reduced_last',
'device_operating_system_user_agent_reduced_first',
'device_operating_system_user_agent_reduced_last',
'device_browser_user_agent_reduced_first',
'device_browser_user_agent_reduced_last']
dummies = pd.get_dummies(df.loc[:, df.columns.isin(categorical_features)], drop_first=True)
df.drop(categorical_features, axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

# create bins for user age
df['user_age_14-25'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1991) & (x <= 2002) else 0)
df['user_age_26-35'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1981) & (x <= 1990) else 0)
df['user_age_36-45'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1971) & (x <= 1980) else 0)
df['user_age_46-55'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1961) & (x <= 1970) else 0)
df['user_age_56-65'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1951) & (x <= 1960) else 0)
df['user_age_65_plus'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1900) & (x <= 1950) else 0)

# flag to indicate visit from Switzerland (add more countries?)
df['Switzerland_first'] = df['country_first'].apply(lambda x: 1 if x == 'Switzerland' else 0)
df['Switzerland_last'] = df['country_last'].apply(lambda x: 1 if x == 'Switzerland' else 0)

print('Processing categorical features complete')



### ADD TIME FEATURES
print('Starting processing time features...')

# visit duration in seconds
df['visit_duration_seconds'] = df['hit_time_gmt_max'] - df['hit_time_gmt_min']
df['visit_duration_seconds'] = df['visit_duration_seconds'].apply(lambda x: x.seconds)

# features for month, day of month, day of week and hour of day
df['month'] = df['date_time_min'].apply(lambda x: x.month)
df['day_of_month'] = df['date_time_min'].apply(lambda x: str(x.day)) # beginning, middle, end
df['day_of_week'] = df['date_time_min'].apply(lambda x: str(x.weekday())) # weekday, weekend
df['hour_of_day'] = df['date_time_min'].apply(lambda x: str(x.hour)) # morning (6-11), noon (11-14), afternoon (14-17), evening (17-23), night (23-6)

# encode time features
time_features = ['month',
'day_of_month',
'day_of_week',
'hour_of_day']
time_dummies = pd.get_dummies(df.loc[:, df.columns.isin(time_features)], drop_first=True)
df.drop(time_features, axis=1, inplace=True)
df = pd.concat([df, time_dummies], axis=1)



# sort dataframe by visitor_id, visit_num and visit_start_time_gmt
df = df.sort_values(['visitor_id', 'visit_num', 'visit_start_time_gmt'], ascending=[True, True, True])

# hours and days since last visit
df['visitor_id_lag'] = df['visitor_id'].shift(1)
df['visit_start_time_gmt_lag'] = df['visit_start_time_gmt'].shift(1)
df['visit_start_time_gmt_minus_visit_start_time_gmt_lag'] = df['visit_start_time_gmt'] - df['visit_start_time_gmt_lag']
df['days_since_last_visit'] = df.apply(lambda x: x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'].days if (pd.notnull(x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'])) & (x['visitor_id'] == x['visitor_id_lag']) else np.nan, axis=1)
df['hours_since_last_visit'] = df.apply(lambda x: x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'].seconds // 3600 if (pd.notnull(x['visit_start_time_gmt_minus_visit_start_time_gmt_lag'])) & (x['visitor_id'] == x['visitor_id_lag']) else np.nan, axis=1)

# visit in last n hours and last n days (correlated with hourly, daily, weekly, monthly, quarterly, yearly visitor?)
df['visit_in_last_12_hours'] = df['hours_since_last_visit'].apply(lambda x: 1 if x <= 12 else 0)
df['visit_in_last_24_hours'] = df['hours_since_last_visit'].apply(lambda x: 1 if x <= 24 else 0)
df['visit_in_last_7_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x >= 1) & (x <= 7) else 0)
df['visit_in_last_14_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x > 7) & (x <= 14) else 0)
df['visit_in_last_21_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x > 14) & (x <= 21) else 0)
df['visit_in_last_28_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x > 21) & (x <= 28) else 0)
df['visit_in_last_28_plus_days'] = df['days_since_last_visit'].apply(lambda x: 1 if x > 28 else 0)

# hours and days since last purchase
purchases_visits.rename(columns={'visit_start_time_gmt' : 'visit_time'}, inplace=True)
purchases_visits['visit_time_minus_purchase_time'] = purchases_visits['visit_time'] - purchases_visits['purchase_time']
purchases_visits['visit_time_minus_purchase_time_delta_hours'] = purchases_visits['purchase_time_minus_visit_time'].apply(lambda x: x.total_seconds() // 3600)
purchases_visits['visit_time_minus_purchase_time_delta_days'] = purchases_visits['purchase_time_minus_visit_time'].apply(lambda x: x.days)

# purchase in last n hours and last n days
purchases_visits['purchase_in_last_12_hours'] = purchases_visits['visit_time_minus_purchase_time_delta_hours'].apply(lambda x: 1 if (x <= 12) & (x >= 0) else 0)
purchases_visits['purchase_in_last_24_hours'] = purchases_visits['visit_time_minus_purchase_time_delta_hours'].apply(lambda x: 1 if (x <= 24) & (x >= 0) else 0)
purchases_visits['purchase_in_last_7_days'] = purchases_visits['visit_time_minus_purchase_time_delta_days'].apply(lambda x: 1 if (x >= 1) & (x <= 7) else 0)
purchases_visits['purchase_in_last_14_days'] = purchases_visits['visit_time_minus_purchase_time_delta_days'].apply(lambda x: 1 if (x > 7) & (x <= 14) else 0)
purchases_visits['purchase_in_last_21_days'] = purchases_visits['visit_time_minus_purchase_time_delta_days'].apply(lambda x: 1 if (x > 14) & (x <= 21) else 0)
purchases_visits['purchase_in_last_28_days'] = purchases_visits['visit_time_minus_purchase_time_delta_days'].apply(lambda x: 1 if (x > 21) & (x <= 28) else 0)
purchases_visits['purchase_in_last_28_plus_days'] = purchases_visits['visit_time_minus_purchase_time_delta_days'].apply(lambda x: 1 if x > 28 else 0)

purchases_visits.rename(columns={'visit_time' : 'visit_start_time_gmt'}, inplace=True)
purchase_in_last_12_hours = purchases_visits[purchases_visits['purchase_in_last_12_hours'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_12_hours']]
purchase_in_last_12_hours.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_12_hours, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_12_hours'].fillna(0, inplace=True)
df['purchase_in_last_12_hours'] = df['purchase_in_last_12_hours'].astype(np.int64)

purchase_in_last_24_hours = purchases_visits[purchases_visits['purchase_in_last_24_hours'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_24_hours']]
purchase_in_last_24_hours.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_24_hours, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_24_hours'].fillna(0, inplace=True)
df['purchase_in_last_24_hours'] = df['purchase_in_last_24_hours'].astype(np.int64)

purchase_in_last_7_days = purchases_visits[purchases_visits['purchase_in_last_7_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_7_days']]
purchase_in_last_7_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_7_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_7_days'].fillna(0, inplace=True)
df['purchase_in_last_7_days'] = df['purchase_in_last_7_days'].astype(np.int64)

purchase_in_last_14_days = purchases_visits[purchases_visits['purchase_in_last_14_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_14_days']]
purchase_in_last_14_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_14_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_14_days'].fillna(0, inplace=True)
df['purchase_in_last_14_days'] = df['purchase_in_last_14_days'].astype(np.int64)

purchase_in_last_21_days = purchases_visits[purchases_visits['purchase_in_last_21_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_21_days']]
purchase_in_last_21_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_21_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_21_days'].fillna(0, inplace=True)
df['purchase_in_last_21_days'] = df['purchase_in_last_21_days'].astype(np.int64)

purchase_in_last_28_days = purchases_visits[purchases_visits['purchase_in_last_28_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_28_days']]
purchase_in_last_28_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_28_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_28_days'].fillna(0, inplace=True)
df['purchase_in_last_28_days'] = df['purchase_in_last_28_days'].astype(np.int64)

purchase_in_last_28_plus_days = purchases_visits[purchases_visits['purchase_in_last_28_plus_days'] == 1][['visitor_id', 'visit_start_time_gmt', 'purchase_in_last_28_plus_days']]
purchase_in_last_28_plus_days.drop_duplicates(subset=['visitor_id', 'visit_start_time_gmt'], keep=False, inplace=True)
df = pd.merge(df, purchase_in_last_28_plus_days, how='left', on=['visitor_id', 'visit_start_time_gmt'])
df['purchase_in_last_28_plus_days'].fillna(0, inplace=True)
df['purchase_in_last_28_plus_days'] = df['purchase_in_last_28_plus_days'].astype(np.int64)



# visit number in current period
#df['visit_num_period'] = df.groupby('visitor_id').cumcount()
#df['visit_num_period'] = df['visit_num_period'] + 1

# purchase number in current period
#purchases['purchase_num_period'] = purchases.groupby('visitor_id').cumcount()
#purchases['purchase_num_period'] = purchases['purchase_num_period'] + 1

# days since first visit
# days since first purchase

# see literature for more feature ideas



# drop columns not needed anymore
columns_to_drop = ['hit_time_gmt_min',
'hit_time_gmt_max',
'date_time_min',
'date_time_max',
'visitor_id_lead', 
'purchase_within_current_visit_lead', 
'user_age_first', 
'user_age_last', 
'country_first', 
'country_last', 
'visitor_id_lag', 
'visit_start_time_gmt_lag', 
'visit_start_time_gmt_minus_visit_start_time_gmt_lag', 
'days_since_last_visit', 
'hours_since_last_visit']
df.drop(columns_to_drop, axis=1, inplace=True)

print('Processing time features complete')



# write data
print('Starting writing data...')

df.to_pickle('../data/processed_data/'+output_file, compression='gzip')

print('Writing data complete.')



# save run time
print('Processing target and features complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'processing_target_and_features_run_time.txt'
run_time_dict = {'processing target and features run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)