#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### PREPROCESSING #####


# In[2]:


print('Starting preprocessing...')


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


input_file = '3_day_sample_aggregated.tsv.gz'
output_file = '3_day_sample_preprocessed.tsv.gz'
#input_data = '6_week_sample_aggregated.tsv.gz'
#output_data = '6_week_sample_preprocessed.tsv.gz'
#input_data = '12_week_sample_aggregated.tsv.gz'
#output_data = '12_week_sample_preprocessed.tsv.gz'
#input_data = '25_week_sample_aggregated.tsv.gz'
#output_data = '25_week_sample_preprocessed.tsv.gz'

print('Input file selected: ', input_file)
print('Output file selected', output_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt', 'last_hit_time_gmt_visit'])

print('Loading data complete.')


# In[8]:


### ENCODE TARGET
print('Encoding target...')


# In[9]:


# binary encode target to be either 1 or 0
df['purchase'] = df['purchase'].apply(lambda x: 1 if x >= 1 else 0)

print('Encoding target complete.')


# In[10]:


### ENCODE CATEGORICAL FEATURES
print('Encoding categorical features...')


# In[11]:


categorical_cols = ['connection_type',
                    'browser',
                    'operating_system_generalized',
                    'search_engine_generalized',
                    'marketing_channel',
                    'referrer_type']

dummies = pd.get_dummies(df.loc[:, df.columns.isin(categorical_cols)], drop_first=True)
df.drop(categorical_cols, axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)


# In[12]:


### note on columns
# cart_open not filled
# static columns: registration_(any_form)_(e20), newsletter_signup_(any_form)_(e26), newsletter_subscriber_(e27), login_success_(e72), logout_success_(e73), login_fail_(e74), registration_fail_(e75)
# columns with lots of missing values: net_promoter_score_raw_(v10)_-_user, user_gender_(v61), user_age_(v62)
# unclear use: event level columns, post_channel (contains 'Order Confirmation')
# hit_of_logged_in_user_(e23) and login_status potentially duplicates


# In[13]:


### WRITE DATAFRAME TO FILE
print('Writing dataframe to file...')


# In[14]:


df.to_csv('../data/processed_data/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[15]:


print('Preprocessing complete.')
print('Run time: ', datetime.now() - start_time)

