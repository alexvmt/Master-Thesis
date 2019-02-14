#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### DATA CLEANING AND PROCESSING #####


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


### get column headers and load data into a dataframe
column_headers = pd.read_csv('../data/mapping_files/column_headers.tsv', sep='\t')
df = pd.read_csv('../data/raw_data/3_day_sample_raw.tsv.gz', compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, names=column_headers)


# In[6]:


print('Time passed since start: ', datetime.now() - start_time)


# In[7]:


##### CLEAN DATA
print('Cleaning data...')


# In[8]:


### drop unnecessary rows
# reset index to make sure that index values are unique
df = df.reset_index(drop=True)

# drop rows where exclude_hit > 1
df = df.drop(df[df.exclude_hit > 0].index)

# drop rows where hit_source is 5, 7, 8 or 9
df = df.drop(df[(df.hit_source == 5) | (df.hit_source == 7) | (df.hit_source == 8) | (df.hit_source == 9)].index)


# In[9]:


### browser mapping
# load file for browser mapping
browser_mapping = pd.read_csv('../data/mapping_files/browser_type.tsv', sep='\t', header=None)
browser_mapping.columns = ['browser_id', 'browser_name']

# create dictionary for browser mapping
browser_mapping_dict = dict(zip(browser_mapping.browser_id, browser_mapping.browser_name))

# map browsers
df['browser'] = df['browser'].map(browser_mapping_dict).fillna(df['browser'])
df['browser'] = df['browser'].apply(lambda x: 'Not Specified' if x == 0 else x)


# In[10]:


### connection type mapping
# load file for connection type mapping
connection_type_mapping = pd.read_csv('../data/mapping_files/connection_type.tsv', sep='\t', header=None)
connection_type_mapping.columns = ['connection_type_id', 'connection_type_name']

# create dictionary for connection type mapping
connection_type_mapping_dict = dict(zip(connection_type_mapping.connection_type_id, connection_type_mapping.connection_type_name))

# map connection types
df['connection_type'] = df['connection_type'].map(connection_type_mapping_dict).fillna(df['connection_type'])


# In[11]:


### country mapping
# load file for country mapping
country_mapping = pd.read_csv('../data/mapping_files/country.tsv', sep='\t', header=None)
country_mapping.columns = ['country_id', 'country_name']

# drop dupliate countries
country_mapping = country_mapping.drop_duplicates('country_name').reset_index(drop=True)

# create dictionary for country mapping
country_mapping_dict = dict(zip(country_mapping.country_id, country_mapping.country_name))

# map countries
df['country'] = df['country'].map(country_mapping_dict).fillna(df['country'])


# In[12]:


### custom evars mapping
# load file for custom evars mapping
evars = pd.read_csv('../data/mapping_files/custom_evars.tsv', sep='\t')
evars_mapping = evars[['id', 'name']]

# map custom evars
evar_cols = [x for x in df.columns if x.lower()[:9] == 'post_evar']
evar_cols = [x.replace('post_', '') for x in evar_cols]
evars_mapped = evars[evars['id'].isin(evar_cols)][['id', 'name']]
evars_mapped['id'] = evars_mapped['id'].apply(lambda x: 'post_' + x)
evars_mapped = evars_mapped.reset_index(drop=True)

# rename custom evars
for i in range(evars_mapped.shape[0]):
    df.rename(columns={evars_mapped.iloc[i,0] : str.lower(evars_mapped.iloc[i,1]).replace(' ','_')}, inplace=True)


# In[13]:


### custom marketing channel mapping
# load file for marketing channel mapping
marketing_channel_mapping = pd.read_csv('../data/mapping_files/custom_marketing_channels.tsv', sep='\t')

# create dictionary for marketing channel mapping
marketing_channel_mapping_dict = dict(zip(marketing_channel_mapping.channel_id, marketing_channel_mapping.name))

# map marketing channels
df['marketing_channel'] = df['va_closer_id'].map(marketing_channel_mapping_dict).fillna(df['va_closer_id'])
df['marketing_channel'] = df['marketing_channel'].apply(lambda x: 'Not Specified' if x == 0 else x)
df.drop('va_closer_id', axis=1, inplace=True)


# In[14]:


### custom events and standard events mapping
# fill missing values in post_event_list
df['post_event_list'] = df['post_event_list'].fillna('Not Specified')

# load file for event mapping
standard_events = pd.read_csv('../data/mapping_files/event.tsv', sep='\t', header=None)
standard_events.columns = ['event_id', 'event_name']

# load file for custom event mapping
custom_events = pd.read_csv('../data/mapping_files/custom_events.tsv', sep='\t')
custom_events['event_id'] = custom_events.index + 200

# map events and custom events
events = pd.merge(standard_events, custom_events, how='inner', on='event_id')
events_mapping = events[['event_id', 'name']]
events_mapping = events_mapping.reset_index(drop=True)

# create event dummies
for id, event in zip(events_mapping.iloc[:,0], events_mapping.iloc[:,1]):
        df[str.lower(event).replace(' ','_')] = df['post_event_list'].apply(lambda x: 1 if ','+str(id)+',' in x else 0)
        
# drop internal users
df = df.drop(df[df['internal_user_(e30)'] == 1].index)


# In[15]:


### custom props mapping
# load file for custom props mapping
props = pd.read_csv('../data/mapping_files/custom_props.tsv', sep='\t')
props_mapping = props[['id', 'name']]

# map custom evars
prop_cols = [x for x in df.columns if x.lower()[:9] == 'post_prop']
prop_cols = [x.replace('post_', '') for x in prop_cols]
props_mapped = props[props['id'].isin(prop_cols)][['id', 'name']]
props_mapped['id'] = props_mapped['id'].apply(lambda x: 'post_' + x)
props_mapped = props_mapped.reset_index(drop=True)

# rename custom props
for i in range(props_mapped.shape[0]):
    df.rename(columns={props_mapped.iloc[i,0] : str.lower(props_mapped.iloc[i,1]).replace(' ','_')}, inplace=True)


# In[16]:


### operating system mapping
# load file for operating system mapping
operating_system_mapping = pd.read_csv('../data/mapping_files/operating_systems.tsv', sep='\t', header=None)
operating_system_mapping.columns = ['operating_system_id', 'operating_system_name']

# create dictionary for operating system mapping
operating_system_mapping_dict = dict(zip(operating_system_mapping.operating_system_id, operating_system_mapping.operating_system_name))

# map operating systems
df['operating_system'] = df['os'].map(operating_system_mapping_dict).fillna(df['os'])
df.drop('os', axis=1, inplace=True)

# generalize operating system
def generalize_operating_system(row):
    if 'Windows' in row['operating_system']:
        return 'Windows'
    elif 'Linux' in row['operating_system']:
        return 'Linux'
    elif 'Android' in row['operating_system']:
        return 'Android'
    elif 'Mobile iOS' in row['operating_system']:
        return 'Apple'
    elif 'Macintosh' in row['operating_system']:
        return 'Apple'
    elif 'OS X' in row['operating_system']:
        return 'Apple'
    elif 'Not Specified' in row['operating_system']:
        return 'Not Specified'
    else:
        return 'Other'
    
df['operating_system_generalized'] = df.apply(generalize_operating_system, axis=1)
df.drop('operating_system', axis=1, inplace=True)


# In[17]:


### referrer type mapping
# load file for referrer type mapping
referrer_type_mapping = pd.read_csv('../data/mapping_files/referrer_type.tsv', sep='\t', header=None)
referrer_type_mapping.columns = ['referrer_type_id', 'referrer_type_name', 'referrer_type']

# create dictionary for referrer type mapping
referrer_type_mapping_dict = dict(zip(referrer_type_mapping.referrer_type_id, referrer_type_mapping.referrer_type))

# map referrer types
df['referrer_type'] = df['ref_type'].map(referrer_type_mapping_dict).fillna(df['ref_type'])
df.drop('ref_type', axis=1, inplace=True)


# In[18]:


### search engine mapping
# load file for search engine mapping
search_engine_mapping = pd.read_csv('../data/mapping_files/search_engines.tsv', sep='\t', header=None)
search_engine_mapping.columns = ['search_engine_id', 'search_engine_name']

# create dictionary for search engine mapping
search_engine_mapping_dict = dict(zip(search_engine_mapping.search_engine_id, search_engine_mapping.search_engine_name))

# map search engines
df['search_engine'] = df['post_search_engine'].map(search_engine_mapping_dict).fillna(df['post_search_engine'])
df.drop('post_search_engine', axis=1, inplace=True)

# convert search engine column to string type
df['search_engine'] = df['search_engine'].astype(str)

# generalize search engine
def generalize_search_engine(row):
    if 'Google' in row['search_engine']:
        return 'Google'
    elif 'Yahoo' in row['search_engine']:
        return 'Yahoo'
    elif 'Bing' in row['search_engine']:
        return 'Bing'
    elif 'Baidu' in row['search_engine']:
        return 'Baidu'
    elif 'DuckDuckGo' in row['search_engine']:
        return 'DuckDuckGo'
    elif 'Yandex' in row['search_engine']:
        return 'Yandex'
    elif 'Search.ch' in row['search_engine']:
        return 'Search.ch'
    elif '0' in row['search_engine']:
        return ' Not Specified'
    else:
        return 'Other'
    
df['search_engine_generalized'] = df.apply(generalize_search_engine, axis=1)
df.drop('search_engine', axis=1, inplace=True)


# In[19]:


### split and process product_items, product_item_price and product_categories columns
df['num_product_items_seen'] = df['product_items'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(';')))

df['sum_price_product_items_seen'] = df['product_item_price'].apply(lambda x: 0 if pd.isnull(x)
                                                              else sum([float(i) for i in x.split(';')]))

df['product_categories_level_1'] = df['product_categories'].apply(lambda x: 'Not Specified' if pd.isnull(x)
                                                                  else [i.split(' / ') for i in x.split(';')][0][0])

df['product_categories_level_2'] = df['product_categories'].apply(lambda x: 'Not Specified' if pd.isnull(x) else 
                                                                  ([i.split(' / ') for i in x.split(';')][0][1] if len([i.split(' / ') for i in x.split(';')][0]) > 1 
                                                                  else 'Not Specified'))

df['product_categories_level_3'] = df['product_categories'].apply(lambda x: 'Not Specified' if pd.isnull(x) else 
                                                                  ([i.split(' / ') for i in x.split(';')][0][2] if len([i.split(' / ') for i in x.split(';')][0]) > 2 
                                                                  else 'Not Specified'))


# In[20]:


print('Time passed since start: ', datetime.now() - start_time)


# In[21]:


##### AGGREGATE NUMERICAL COLUMNS TO SESSION LEVEL
print('Aggregating numerical columns...')


# In[22]:


### aggregate numerical columns
# convert hit_time_gmt column from unix format to datetime format
df['hit_time_gmt'] = pd.to_datetime(df['hit_time_gmt'], unit='s')

# convert visit_page_num column to integer type
df['visit_page_num'] = df['visit_page_num'].astype(np.int64)

# fill missing values with 0 in cart_value_(v50) column
df['cart_value_(v50)'].fillna(0, inplace=True)

# select numerical columns
numerical_cols_names = ['visitor_id', 'visit_num', 'visit_page_num', 'hit_time_gmt' ,'purchase_boolean', 
                        'product_view_boolean', 'checkout_boolean', 'cart_addition_boolean', 'cart_removal_boolean', 
                        'cart_view_boolean', 'campaign_view_boolean', 'cart_value_(v50)', 'page_view_boolean', 
                        'last_purchase_num', 'num_product_items_seen', 'sum_price_product_items_seen']
numerical_cols = df.loc[:, df.columns.isin(numerical_cols_names)].copy()

# group by visitor_id and visit_num and aggregate numerical columns
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
                                                                                                          'sum_price_product_items_seen' : 'sum'})
numerical_cols_aggregated.columns = list(numerical_cols_aggregated.columns)
numerical_cols_aggregated.columns = ['visitor_id', 'visit_num', 'visit_page_num', 'hit_time_gmt', 'last_hit_time_gmt_visit', 
                                     'purchase', 'product_view', 'checkout', 'cart_addition', 'cart_removal', 'cart_view', 
                                     'campaign_view', 'cart_value', 'page_view', 'last_purchase_num', 
                                     'num_product_items_seen', 'sum_price_product_items_seen']

# sort dataframe by hit_time_gmt, last_hit_time_gmt_visit, visitor_id and visit_num
numerical_cols_aggregated = numerical_cols_aggregated.sort_values(['hit_time_gmt', 'last_hit_time_gmt_visit', 'visitor_id', 'visit_num'], ascending=[True, True, True, True])
numerical_cols_aggregated = numerical_cols_aggregated.reset_index(drop=True)


# In[82]:


### ensure correct dtypes
object_cols = ['visitor_id']

for i in object_cols:
    numerical_cols_aggregated[i] = numerical_cols_aggregated[i].astype(str)

int_cols = ['visit_num',
 'visit_page_num',
 'purchase',
 'product_view',
 'checkout',
 'cart_addition',
 'cart_removal',
 'cart_view',
 'campaign_view',
 'page_view',
 'last_purchase_num',
 'num_product_items_seen']

for i in int_cols:
    numerical_cols_aggregated[i] = numerical_cols_aggregated[i].astype(np.int64)
    
float_cols = ['cart_value', 'sum_price_product_items_seen']

for i in float_cols:
    numerical_cols_aggregated[i] = numerical_cols_aggregated[i].astype(np.float64)

datetime_cols = ['hit_time_gmt', 'last_hit_time_gmt_visit']

for i in datetime_cols:
    numerical_cols_aggregated[i] = numerical_cols_aggregated[i].astype(np.datetime64)


# In[23]:


print('Time passed since start: ', datetime.now() - start_time)


# In[24]:


##### PROCESS CATEGORICAL COLUMNS
print('Processing categorical columns...')


# In[25]:


# select categorical columns
categorical_cols_names = ['visitor_id', 'hit_time_gmt', 'connection_type', 'country', 'geo_city', 'geo_dma', 
                          'geo_region', 'geo_zip', 'post_channel', 'post_cookies', 'net_promoter_score_raw_(v10)_-_user', 
                          'registered_user_(user)_(v34)', 'login_status_(hit)_(v37)', 'user_gender_(v61)', 'user_age_(v62)',
                          'post_persistent_cookie', 'browser_generalized', 'operating_system_generalized',
                          'search_engine_generalized', 'marketing_channel', 'referrer_type', 'hit_of_logged_in_user_(e23)',
                          'visit_during_tv_spot_(e71)', 'repeat_orders_(e9)', 'newsletter_subscriber_(e27)', 
                          'registration_(any_form)_(e20)', 'registration_fail_(e75)', 'newsletter_signup_(any_form)_(e26)', 
                          'new_visit', 'hourly_visitor', 'daily_visitor', 'weekly_visitor', 'monthly_visitor', 
                          'quarterly_visitor', 'yearly_visitor', 'product_categories_level_1', 'product_categories_level_2',
                          'product_categories_level_3']
categorical_cols = df.loc[:, df.columns.isin(categorical_cols_names)].copy()

# transform categorical columns where necessary
categorical_cols['post_cookies'] = categorical_cols['post_cookies'].apply(lambda x: 1 if x == 'Y' else 0)
categorical_cols['registered_user'] = categorical_cols['registered_user_(user)_(v34)'].apply(lambda x: 1 if x == 'y' else 0)
categorical_cols.drop('registered_user_(user)_(v34)', axis=1, inplace=True)
categorical_cols['login_status'] = categorical_cols['login_status_(hit)_(v37)'].apply(lambda x: 1 if x == 'y' else 0)
categorical_cols.drop('login_status_(hit)_(v37)', axis=1, inplace=True)
categorical_cols['post_persistent_cookie'] = categorical_cols['post_persistent_cookie'].apply(lambda x: 1 if x == 'Y' else 0)
categorical_cols['geo_city'] = categorical_cols['geo_city'].apply(lambda x: 'Not Specified' if x == '?' else x)
categorical_cols['geo_region'] = categorical_cols['geo_region'].apply(lambda x: 'Not Specified' if x == '?' else x)

# static columns: registration_(any_form)_(e20), newsletter_signup_(any_form)_(e26), newsletter_subscriber_(e27), registration_fail_(e75)
# columns with lots of missing values: NPS, gender, age

# sort dataframe by hit_time_gmt_min, visitor_id and visit_num
categorical_cols = categorical_cols.sort_values(['hit_time_gmt', 'visitor_id'], ascending=[True, True])
categorical_cols = categorical_cols.reset_index(drop=True)


# In[84]:


### ensure correct dtypes
object_cols = ['connection_type',
 'country',
 'geo_city',
 'geo_region',
 'geo_zip',
 'post_channel',
 'user_gender_(v61)',
 'user_age_(v62)',
 'net_promoter_score_raw_(v10)_-_user',
 'visitor_id',
 'marketing_channel',
 'operating_system_generalized',
 'referrer_type',
 'search_engine_generalized',
 'product_categories_level_1',
 'product_categories_level_2',
 'product_categories_level_3']

for i in object_cols:
    categorical_cols[i] = categorical_cols[i].astype(str)

int_cols = ['daily_visitor',
 'geo_dma',
 'hourly_visitor',
 'monthly_visitor',
 'new_visit',
 'post_cookies',
 'post_persistent_cookie',
 'quarterly_visitor',
 'weekly_visitor',
 'yearly_visitor',
 'repeat_orders_(e9)',
 'registration_(any_form)_(e20)',
 'hit_of_logged_in_user_(e23)',
 'newsletter_signup_(any_form)_(e26)',
 'newsletter_subscriber_(e27)',
 'visit_during_tv_spot_(e71)',
 'registration_fail_(e75)',
 'registered_user',
 'login_status']

for i in int_cols:
    categorical_cols[i] = categorical_cols[i].astype(np.int64)

datetime_cols = ['hit_time_gmt']

for i in datetime_cols:
    categorical_cols[i] = categorical_cols[i].astype(np.datetime64)


# In[26]:


print('Time passed since start: ', datetime.now() - start_time)


# In[27]:


##### MERGE NUMERICAL AND CATEGORICAL COLUMNS ON SESSION LEVEL
print('Merging numerical and categorical columns...')


# In[28]:


session_level_data_merged = pd.merge_asof(numerical_cols_aggregated, categorical_cols, on='hit_time_gmt', by='visitor_id')
session_level_data_merged = session_level_data_merged.reset_index(drop=True)


# In[29]:


print('Time passed since start: ', datetime.now() - start_time)


# In[30]:


##### WRITE DATA TO FILE
print('Writing data to file...')


# In[31]:


session_level_data_merged.to_csv('../data/processed_data/session_level_data_merged.tsv.gz', compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[32]:


print('Total execution time: ', datetime.now() - start_time)

