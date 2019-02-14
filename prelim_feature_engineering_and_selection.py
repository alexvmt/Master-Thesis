#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### PRELIMINARY FEATURE ENGINEERING AND SELECTION #####


# In[2]:


### import libraries
import pandas as pd
import numpy as np
from datetime import datetime,date


# In[3]:


start_time = datetime.now()
print('Start time: ', start_time)


# In[4]:


##### LOAD DATA
print('Loading data...')


# In[5]:


df = pd.read_csv('../data/processed_data/session_level_data_merged.tsv.gz', compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt', 'last_hit_time_gmt_visit'])


# In[6]:


print('Time passed since start: ', datetime.now() - start_time)


# In[7]:


### ENCODE TARGET
print('Encoding target...')


# In[8]:


# binary encode target to be either 1 or 0
df['purchase'] = df['purchase'].apply(lambda x: 1 if x >= 1 else 0)


# In[9]:


print('Time passed since start: ', datetime.now() - start_time)


# In[10]:


### ADD FEATURES
print('Adding features...')


# In[11]:


### add time features
# sort dataframe by visitor_id, visit_num, hit_time_gmt and last_hit_time_gmt_visit
df = df.sort_values(['visitor_id', 'visit_num', 'hit_time_gmt', 'last_hit_time_gmt_visit'], ascending=[True, True, True, True])

# day of week
df['day_of_week'] = df['hit_time_gmt'].dt.dayofweek

# hour of day
df['hour_of_day'] = df['hit_time_gmt'].dt.hour

# calculate visit duration in seconds
df['visit_duration_seconds'] = df['last_hit_time_gmt_visit'] - df['hit_time_gmt']
df['visit_duration_seconds'] = df['visit_duration_seconds'].apply(lambda x: x.seconds)

# add lag columns for visitor_id and last_hit_time_gmt_visit
df['visitor_id_lag'] = df['visitor_id'].shift(1)
df['last_hit_time_gmt_visit_lag'] = df['last_hit_time_gmt_visit'].shift(1)

# calculate days since last visit and add flag for visit in last 7 days
df['days_since_last_visit'] = df.apply(lambda x: x['hit_time_gmt'] - x['last_hit_time_gmt_visit_lag'] 
                                       if x['visitor_id'] == x['visitor_id_lag'] 
                                       else np.nan, axis=1)
df['days_since_last_visit'] = df['days_since_last_visit'].apply(lambda x: x.days)
df['visit_in_last_7_days'] = df['days_since_last_visit'].apply(lambda x: 1 if (x >=0) & (x <= 7) else 0)

# calculate days since last purchase and add flag for purchase in last 7 days
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


# In[12]:


# add visit number in the current period
df['visit_num_period'] = df.groupby('visitor_id').cumcount()
df['visit_num_period'] = df['visit_num_period'] + 1


# In[13]:


# add flag to indicate bounce
df['bounce'] = df['visit_page_num'].apply(lambda x: 1 if x == 1 else 0)


# In[14]:


# add flag to indicate visit from Switzerland
df['Switzerland'] = df['country'].apply(lambda x: 1 if x == 'Switzerland' else 0)


# In[15]:


print('Time passed since start: ', datetime.now() - start_time)


# In[16]:


### ENCODE CATEGORICAL FEATURES
print('Encoding categorical features...')


# In[17]:


### encode categorical features
df.drop(['visitor_id', 
         'visitor_id_lag', 
         'last_hit_time_gmt_visit', 
         'last_hit_time_gmt_visit_lag',
         'days_since_last_visit',
         'days_since_last_purchase',
         # temporarily drop columns where use is unclear or they have many missing values
         'country', 
         'geo_city',
         'geo_region',
         'geo_zip',
         'user_gender_(v61)',
         'net_promoter_score_raw_(v10)_-_user',
         'product_categories_level_1',
         'product_categories_level_2',
         'product_categories_level_3',
         'post_channel'], axis=1, inplace=True)
object_cols = list(df.select_dtypes(include=['object']).columns)
dummies = pd.get_dummies(df.loc[:, df.columns.isin(object_cols)], drop_first=True)
df.drop(object_cols, axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

### note on columns
# cart_open not filled
# post_channel contains 'Order Confirmation'
# static columns: registration_(any_form)_(e20), newsletter_signup_(any_form)_(e26), newsletter_subscriber_(e27), registration_fail_(e75)
# columns with lots of missing values: NPS, gender, age


# In[18]:


print('Time passed since start: ', datetime.now() - start_time)


# In[19]:


### WRITE DATA TO FILE


# In[20]:


df.to_csv('../data/processed_data/session_level_data_final.tsv.gz', compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[21]:


print('Total execution time: ', datetime.now() - start_time)

