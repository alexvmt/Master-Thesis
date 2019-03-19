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

# sort dataframe by visit_start_time_gmt, visitor_id and visit_num
df = df.sort_values(['visit_start_time_gmt', 'visitor_id', 'visit_num'], ascending=[True, True, True])

# create lag and lead columns
df['visitor_id_lag'] = df['visitor_id'].shift(1)
df['visitor_id_lead'] = df['visitor_id'].shift(-1)
df['hit_time_gmt_max_lag'] = df['hit_time_gmt_max'].shift(1)
df['purchase_within_current_visit_lead'] = df['purchase_boolean_sum'].shift(-1)
df['visit_start_time_gmt_lag'] = df['visit_start_time_gmt'].shift(1)



### PROCESS TARGET

# purchase within current visit
df['purchase_within_current_visit'] = df['purchase_boolean_sum'].apply(lambda x: 1 if x >= 1 else 0)

# purchase within next visit (?)
df['purchase_within_current_visit_lead'] = df['purchase_within_current_visit_lead'].apply(lambda x: 1 if x >= 1 else 0)
df['purchase_within_next_visit'] = df.apply(lambda x: 1 if (x['purchase_within_current_visit'] == 1) | ((x['visitor_id'] == x['visitor_id_lead']) & (x['purchase_within_current_visit_lead'] == 1)) else 0, axis=1)

# purchase within next 24 hours
#df['purchase_within_next_24_hours'] = TBD

# purchase within next 7 days
#df['purchase_within_next_7_days'] = TBD



### PROCESS CATEGORICAL FEATURES

# clean and reduce levels of categorical columns if necessary (if 10+ levels drop levels with less than 0,1% of max)
df = process_product_categories(df)
df = process_net_promoter_score(df)
df = process_user_gender(df)
df = process_user_age(df)
df = process_search_engines(df)
df = process_device_types(df)
df = process_device_brand_names(df)
df = process_device_operating_systems(df)
df = process_device_browsers(df)

# encode other categorical columns (correlated and static features, fill gaps between first and last for static features like age or gender)
categorical_columns = ['country_first',
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
dummies = pd.get_dummies(df.loc[:, df.columns.isin(categorical_columns)], drop_first=True)
df = pd.concat([df, dummies], axis=1)

# create bins for user age
df['user_age_14-25'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1991) (x <= 2002) else 0)
df['user_age_26-35'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1981) & (x <= 1990) else 0)
df['user_age_36-45'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1971) & (x <= 1980) else 0)
df['user_age_46-55'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1961) & (x <= 1970) else 0)
df['user_age_56-65'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1951) & (x <= 1960) else 0)
df['user_age_65+'] = df['user_age_first'].apply(lambda x: 1 if (x >= 1900) & (x <= 1950) else 0)



### ADD FEATURES

# features for month, day of month, day of week and hour of day
df['month'] = df['visit_start_time_gmt'].apply(lambda x: x.month)
df['day_of_month'] = df['visit_start_time_gmt'].apply(lambda x: x.day) # beginning, middle, end
df['day_of_week'] = df['visit_start_time_gmt'].apply(lambda x: x.weekday()) # weekday, weekend
df['hour_of_day'] = df['visit_start_time_gmt'].apply(lambda x: x.hour) # morning (6-11), noon (11-14), afternoon (14-17), evening (17-23), night (23-6)

# encode time features
time_features = ['month',
'day_of_month',
'day_of_week',
'hour_of_day']
time_dummies = pd.get_dummies(df.loc[:, df.columns.isin(time_features)], drop_first=True)
df.drop(time_features, axis=1, inplace=True)
df = pd.concat([df, time_dummies], axis=1)

# visit duration in seconds
df['visit_duration_seconds'] = df['hit_time_gmt_max'] - df['visit_start_time_gmt']
df['visit_duration_seconds'] = df['visit_duration_seconds'].apply(lambda x: x.seconds)

# visit number in current period
df['visit_num_period'] = df.groupby('visitor_id').cumcount()
df['visit_num_period'] = df['visit_num_period'] + 1

# flag to indicate visit from Switzerland (add more countries?)
df['Switzerland_first'] = df['country_first'].apply(lambda x: 1 if x == 'Switzerland' else 0)
df['Switzerland_last'] = df['country_last'].apply(lambda x: 1 if x == 'Switzerland' else 0)

# days and hours since last visit TBD
df['days_since_last_visit'] = df.apply(lambda x: x['hit_time_gmt_min'] - x['hit_time_gmt_max_lag'] if x['visitor_id'] == x['visitor_id_lag'] else np.nan, axis=1)
df['days_since_last_visit'] = df['days_since_last_visit'].apply(lambda x: x.days if pd.notnull(x) else x)
df['hours_since_last_visit'] = df['days_since_last_visit'].apply(lambda x: x.seconds//3600 if pd.notnull(x) else x)

# visit in n last days and hours TBD (correlated with hourly, daily, weekly, monthly, quarterly, yearly visitor?)
df['visit_in_last_12_hours'] = df['hours_since_last_visit'].apply(lambda x: 1 if x <= 12 else 0)
df['visit_in_last_24_hours'] = df['hours_since_last_visit'].apply(lambda x: 1 if x <= 24 else 0)
df['visit_in_last_7_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x >= 1) & (x <= 7) else 0)
df['visit_in_last_14_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x > 7) & (x <= 14) else 0)
df['visit_in_last_21_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x > 14) & (x <= 21) else 0)
df['visit_in_last_28_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x > 21) & (x <= 28) else 0)
df['visit_in_last_28+_days'] = df['days_since_last_visit'].apply(lambda x: 1 if x > 28 else 0)



# days since last purchase TBD
purchases = df[df['purchase_within_current_visit'] == 1][['visitor_id', 'visitor_id_lag', 'visit_start_time_gmt', 'visit_start_time_gmt_lag', 'visit_num']]
purchases = purchases.sort_values(['visitor_id', 'visit_start_time_gmt'], ascending=[True, True])
purchases['days_since_last_purchase'] = purchases.apply(lambda x: x['visit_start_time_gmt'] - x['visit_start_time_gmt_lag'] if x['visitor_id'] == x['visitor_id_lag'] else np.nan, axis=1)
purchases['days_since_last_purchase'] = purchases['days_since_last_purchase'].apply(lambda x: x.days)
purchases['hours_since_last_visit'] = purchases['days_since_last_purchase'].apply(lambda x: x.hours)

# purchase in last n days and hours TBD
purchases['purchase_in_last_12_hours'] = purchases['hours_since_last_visit'].apply(lambda x: 1 if x <= 12 else 0)
purchases['purchase_in_last_24_hours'] = purchases['hours_since_last_visit'].apply(lambda x: 1 if x <= 24 else 0)
purchases['purchase_in_last_7_days'] = purchases['days_since_last_purchase'].apply(lambda x: 1 if (x >= 1) & (x <= 7) else 0)
purchases['purchase_in_last_14_days'] = purchases['days_since_last_purchase'].apply(lambda x: 1 if (x > 7) & (x <= 14) else 0)
purchases['purchase_in_last_21_days'] = purchases['days_since_last_purchase'].apply(lambda x: 1 if (x > 14) & (x <= 21) else 0)
purchases['purchase_in_last_28_days'] = purchases['days_since_last_purchase'].apply(lambda x: 1 if (x > 21) & (x <= 28) else 0)
purchases['purchase_in_last_28+_days'] = purchases['days_since_last_purchase'].apply(lambda x: 1 if x > 28 else 0)

# purchase number in current period TBD
purchases['purchase_num_period'] = purchases.groupby('visitor_id').cumcount()
purchases['purchase_num_period'] = purchases['purchase_num_period'] + 1

# days since first visit
# days since first purchase



### SELECT FEATURES
columns_to_drop = []

for col in columns_to_drop:
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)
    else:
        pass
    
    
    
# write data
print('Starting writing data...')

df_processed_final.to_pickle('../data/processed_data/'+output_file+categorical_aggregation_mode, compression='gzip')

print('Writing data complete.')



# save run time
print('Processing target and features complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'processing_target_and_features_run_time.txt'
run_time_dict = {'processing target and features run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)