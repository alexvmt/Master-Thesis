#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### ADDING FEATURES #####


# In[2]:


print('Starting adding features...')


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


input_file = '3_day_sample_preprocessed.tsv.gz'
output_file = '3_day_sample_preprocessed_with_additional_features.tsv.gz'
#input_file = '6_week_sample_preprocessed.tsv.gz'
#output_file = '6_week_sample_preprocessed_with_additional_features.tsv.gz'
#input_file = '12_week_sample_preprocessed.tsv.gz'
#output_file = '12_week_sample_preprocessed_with_additional_features.tsv.gz'
#input_file = '25_week_sample_preprocessed.tsv.gz'
#output_file = '25_week_sample_preprocessed_with_additional_features.tsv.gz'

print('Input file selected: ', input_file)
print('Output file selected', output_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt', 'last_hit_time_gmt_visit', 'date_time'])

print('Loading data complete.')


# In[8]:


### ADD ADDITIONAL FEATURES
print('Adding additional features...')


# In[9]:


# sort dataframe by visitor_id, visit_num, hit_time_gmt and last_hit_time_gmt_visit
df = df.sort_values(['visitor_id', 'visit_num', 'hit_time_gmt', 'last_hit_time_gmt_visit'], ascending=[True, True, True, True])

# day of week
df['day_of_week'] = df['date_time'].apply(lambda x: x.weekday())

# hour of day
df['hour_of_day'] = df['date_time'].apply(lambda x: x.hour)

# encode time features
time_features = ['day_of_week',
                 'hour_of_day']
dummies = pd.get_dummies(df.loc[:, df.columns.isin(time_features)], drop_first=True)
df.drop(time_features, axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)


# In[10]:


# visit duration in seconds
df['visit_duration_seconds'] = df['last_hit_time_gmt_visit'] - df['hit_time_gmt']
df['visit_duration_seconds'] = df['visit_duration_seconds'].apply(lambda x: x.seconds)


# In[11]:


# lag columns for visitor_id and last_hit_time_gmt_visit
df['visitor_id_lag'] = df['visitor_id'].shift(1)
df['last_hit_time_gmt_visit_lag'] = df['last_hit_time_gmt_visit'].shift(1)

# days since last visit and flag for visit in last 7 days
df['days_since_last_visit'] = df.apply(lambda x: x['hit_time_gmt'] - x['last_hit_time_gmt_visit_lag'] 
                                       if x['visitor_id'] == x['visitor_id_lag'] 
                                       else np.nan, axis=1)
df['days_since_last_visit'] = df['days_since_last_visit'].apply(lambda x: x.days)
df['visit_in_last_7_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x >=0) & (x <= 7) else 0)


# In[12]:


# days since last purchase and flag for purchase in last 7 days
df['purchase_date'] = df.apply(lambda x: x['hit_time_gmt'] if x['purchase'] == 1 else np.nan, axis=1)
purchases = df[df['purchase'] == 1][['visitor_id', 'purchase_date']]
purchases = purchases.sort_values(['visitor_id', 'purchase_date'], ascending=[True, True])
purchases['visitor_id_lag'] = purchases['visitor_id'].shift(1)
purchases['purchase_date_lag'] = purchases['purchase_date'].shift(1)

purchases['days_since_last_purchase'] = purchases.apply(lambda x: x['purchase_date'] - x['purchase_date_lag']
                                                  if x['visitor_id'] == x['visitor_id_lag']
                                                  else np.nan, axis=1)
purchases['days_since_last_purchase'] = purchases['days_since_last_purchase'].apply(lambda x: x.days)
purchases['purchase_in_last_7_days'] = purchases['days_since_last_purchase'].apply(lambda x: 1 if (x >=0) & (x <= 7) else 0)
purchases['purchase_num_period'] = purchases.groupby('visitor_id').cumcount()
purchases['purchase_num_period'] = purchases['purchase_num_period'] + 1
purchases.drop('visitor_id_lag', axis=1, inplace=True)

df = pd.merge(df, purchases, on=['visitor_id', 'purchase_date'], how='left')
df['purchase_in_last_7_days'] = df['purchase_in_last_7_days'].fillna(0).astype(np.int64)
df['purchase_num_period'] = df['purchase_num_period'].fillna(0).astype(np.int64)


# In[13]:


# visit number in the current period
df['visit_num_period'] = df.groupby('visitor_id').cumcount()
df['visit_num_period'] = df['visit_num_period'] + 1


# In[14]:


# flag to indicate bounce
df['bounce'] = df['visit_page_num'].apply(lambda x: 1 if x == 1 else 0)


# In[15]:


# flag to indicate visit from Switzerland
df['Switzerland'] = df['country'].apply(lambda x: 1 if x == 'Switzerland' else 0)


# In[16]:


### WRITE DATAFRAME TO FILE
print('Writing dataframe to file...')


# In[17]:


df.to_csv('../data/processed_data/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[18]:


print('Adding additional features complete.')
print('Run time: ', datetime.now() - start_time)

