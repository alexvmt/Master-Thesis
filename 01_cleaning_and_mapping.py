#!/usr/bin/env python
# coding: utf-8

# In[1]:


########## CLEANING AND MAPPING ##########


# In[2]:


print('Starting cleaning and mapping...')


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


input_file = '3_day_sample_raw.tsv.gz'
output_file = '3_day_sample_cleaned_and_mapped.tsv.gz'
#input_file = '6_week_sample_raw.tsv.gz'
#output_file = '6_week_sample_cleaned_and_mapped.tsv.gz'
#input_file = '12_week_sample_raw.tsv.gz'
#output_file = '12_week_sample_cleaned_and_mapped.tsv.gz'
#input_file = '25_week_sample_raw.tsv.gz'
#output_file = '25_week_sample_cleaned_and_mapped.tsv.gz'

print('Input file selected: ', input_file)
print('Output file selected', output_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


column_headers = pd.read_csv('../data/mapping_files/column_headers.tsv', sep='\t')
df = pd.read_csv('../data/raw_data/'+input_file, compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, names=column_headers)

print('Loading data complete.')


# In[8]:


##### CLEAN DATA
print('Cleaning data...')


# In[9]:


# reset index to make sure that index values are unique
df = df.reset_index(drop=True)

# drop rows where exclude_hit > 1
df = df.drop(df[df.exclude_hit > 0].index)

# drop rows where hit_source is 5, 7, 8 or 9
df = df.drop(df[(df.hit_source == 5) | (df.hit_source == 7) | (df.hit_source == 8) | (df.hit_source == 9)].index)

print('Dropping rows complete.')


# In[10]:


### browser mapping
# load file for browser mapping
browser_mapping = pd.read_csv('../data/mapping_files/browser_type.tsv', sep='\t', header=None)
browser_mapping.columns = ['browser_id', 'browser_name']

# create dictionary for browser mapping
browser_mapping_dict = dict(zip(browser_mapping.browser_id, browser_mapping.browser_name))

# map browsers
df['browser'] = df['browser'].map(browser_mapping_dict).fillna(df['browser'])
df['browser'] = df['browser'].apply(lambda x: 'Not Specified' if x == 0 else x)

print('Browser mapping complete.')


# In[11]:


### connection type mapping
# load file for connection type mapping
connection_type_mapping = pd.read_csv('../data/mapping_files/connection_type.tsv', sep='\t', header=None)
connection_type_mapping.columns = ['connection_type_id', 'connection_type_name']

# create dictionary for connection type mapping
connection_type_mapping_dict = dict(zip(connection_type_mapping.connection_type_id, connection_type_mapping.connection_type_name))

# map connection types
df['connection_type'] = df['connection_type'].map(connection_type_mapping_dict).fillna(df['connection_type'])

print('Connection type mapping complete.')


# In[12]:


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

print('Country mapping complete.')


# In[13]:


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
    
print('Custom evars mapping complete.')


# In[14]:


### custom marketing channel mapping
# load file for marketing channel mapping
marketing_channel_mapping = pd.read_csv('../data/mapping_files/custom_marketing_channels.tsv', sep='\t')

# create dictionary for marketing channel mapping
marketing_channel_mapping_dict = dict(zip(marketing_channel_mapping.channel_id, marketing_channel_mapping.name))

# map marketing channels
df['marketing_channel'] = df['va_closer_id'].map(marketing_channel_mapping_dict).fillna(df['va_closer_id'])
df['marketing_channel'] = df['marketing_channel'].apply(lambda x: 'Not Specified' if x == 0 else x)

print('Custom marketing channel mapping complete.')


# In[15]:


### standard and custom events mapping
# fill missing values in post_event_list
df['post_event_list'] = df['post_event_list'].fillna('Not Specified')

# load file for standard event mapping
standard_events = pd.read_csv('../data/mapping_files/event.tsv', sep='\t', header=None)
standard_events.columns = ['event_id', 'event_name']

# load file for custom event mapping
custom_events = pd.read_csv('../data/mapping_files/custom_events.tsv', sep='\t')
custom_events['event_id'] = custom_events.index + 200

# map standard and custom events
events = pd.merge(standard_events, custom_events, how='inner', on='event_id')
events_mapping = events[['event_id', 'name']]
events_mapping = events_mapping.reset_index(drop=True)

# create event dummies
for id, event in zip(events_mapping.iloc[:,0], events_mapping.iloc[:,1]):
        df[str.lower(event).replace(' ','_')] = df['post_event_list'].apply(lambda x: 1 if ','+str(id)+',' in x else 0)
        
# drop internal users
df = df.drop(df[df['internal_user_(e30)'] == 1].index)

print('Standard and custom events mapping complete.')


# In[16]:


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
    
print('Custom props mapping complete.')


# In[17]:


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

print('Operating system mapping complete.')


# In[18]:


### referrer type mapping
# load file for referrer type mapping
referrer_type_mapping = pd.read_csv('../data/mapping_files/referrer_type.tsv', sep='\t', header=None)
referrer_type_mapping.columns = ['referrer_type_id', 'referrer_type_name', 'referrer_type']

# create dictionary for referrer type mapping
referrer_type_mapping_dict = dict(zip(referrer_type_mapping.referrer_type_id, referrer_type_mapping.referrer_type))

# map referrer types
df['referrer_type'] = df['ref_type'].map(referrer_type_mapping_dict).fillna(df['ref_type'])

print('Referrer type mapping complete.')


# In[19]:


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

print('Search engine mapping complete.')


# In[20]:


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

print('Product item, prices and categories splitting complete.')


# In[21]:


### filling missing and faulty values
df['cart_value_(v50)'].fillna(0, inplace=True)
df['geo_city'] = df['geo_city'].apply(lambda x: 'Not Specified' if x == '?' else x)
df['geo_region'] = df['geo_region'].apply(lambda x: 'Not Specified' if x == '?' else x)
df['post_channel'] = df['post_channel'].fillna('Not Specified')
df['post_cookies'] = df['post_cookies'].apply(lambda x: 1 if x == 'Y' else 0)
df['post_persistent_cookie'] = df['post_persistent_cookie'].apply(lambda x: 1 if x == 'Y' else 0)
df['registered_user'] = df['registered_user_(user)_(v34)'].apply(lambda x: 1 if x == 'y' else 0)
df['login_status'] = df['login_status_(hit)_(v37)'].apply(lambda x: 1 if x == 'y' else 0)

print('Filling missing and faulty values complete.')


# In[22]:


### casting data types
df['hit_time_gmt'] = pd.to_datetime(df['hit_time_gmt'], unit='s')
df['date_time'] = df['date_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df['visit_page_num'] = df['visit_page_num'].astype(np.int64)

print('Casting data types complete.')


# In[23]:


### drop some columns
columns_to_keep = ['visitor_id', 
                   'hit_time_gmt',
                   'date_time',
                   # numerical columns
                   'visit_num', 
                   'visit_page_num', 
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
                   'sugg._search_result_clicked_(e18)',
                   # categorical columns
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

df = df[columns_to_keep]

### note on columns
# cart_open not filled
# static columns: registration_(any_form)_(e20), newsletter_signup_(any_form)_(e26), newsletter_subscriber_(e27), login_success_(e72), logout_success_(e73), login_fail_(e74), registration_fail_(e75)
# columns with lots of missing values: net_promoter_score_raw_(v10)_-_user, user_gender_(v61), user_age_(v62)
# unclear use: event level columns, post_channel (contains 'Order Confirmation'), last_purchase_num
# hit_of_logged_in_user_(e23) and login_status potentially duplicates

print('Dropping columns complete.')


# In[24]:


##### WRITE DATAFRAME TO FILE
print('Writing dataframe to file...')


# In[25]:


df.to_csv('../data/processed_data/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[26]:


print('Cleaning and mapping complete.')
print('Run time: ', datetime.now() - start_time)

