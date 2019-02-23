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
#input_file = '6_week_sample_aggregated.tsv.gz'
#output_file = '6_week_sample_preprocessed.tsv.gz'
#input_file = '12_week_sample_aggregated.tsv.gz'
#output_file = '12_week_sample_preprocessed.tsv.gz'
#input_file = '25_week_sample_aggregated.tsv.gz'
#output_file = '25_week_sample_preprocessed.tsv.gz'

print('Input file selected: ', input_file)
print('Output file selected', output_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


df = pd.read_csv('../data/processed_data/'+input_file, compression='gzip', sep='\t', low_memory=False, encoding='iso-8859-1', parse_dates=['hit_time_gmt', 'last_hit_time_gmt_visit', 'date_time'])

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


### WRITE DATAFRAME TO FILE
print('Writing dataframe to file...')


# In[13]:


df.to_csv('../data/processed_data/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[14]:


print('Preprocessing complete.')
print('Run time: ', datetime.now() - start_time)

