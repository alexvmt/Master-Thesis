#!/usr/bin/env python
# coding: utf-8

##### HELPER FUNCTIONS #####

# import libraries
import numpy as np
import pandas as pd
from datetime import datetime,date
from device_detector import DeviceDetector
import re



### CLEANING

def drop_rows(df):

    print('Starting dropping rows...')

    # keep rows where exclude_hit is <= 0
    df = df[df['exclude_hit'] <= 0]

    # keep rows where hit_source != 5, 7, 8 or 9
    df = df[(df['hit_source'] != 5) | (df['hit_source'] != 7) |(df['hit_source'] != 8) |(df['hit_source'] != 9)]

    # keep rows where visitor_id is not missing (6 missing values)
    df = df[pd.notnull(df['visitor_id'])]

    # clean visit_page_num and keep rows where visit_page_num is not missing or faulty (118 missing values and 269 faulty values)
    df['visit_page_num'] = df['visit_page_num'].apply(lambda x: np.nan if len(str(x)) > 10 else x)
    df = df[pd.notnull(df['visit_page_num'])]

    print('Dropping rows complete.')

    return df



def drop_columns(df):

    print('Starting dropping columns...')

    # select columns to keep
    columns_to_keep = ['visitor_id',
                       'visit_start_time_gmt',
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
                       'cart_value',
                       'page_view_boolean',
                       'last_purchase_num',
                       'standard_search_results_clicked',
                       'standard_search_started',
                       'suggested_search_results_clicked',
                       # categorical columns
                       'country',
                       'cookies',
                       'persistent_cookie',
                       'search_page_num',
                       'connection_type',
                       'search_engine',
                       'marketing_channel',
                       'referrer_type',
                       'new_visit',
                       'hourly_visitor',
                       'daily_visitor',
                       'weekly_visitor',
                       'monthly_visitor',
                       'quarterly_visitor',
                       'yearly_visitor',
                       'product_items',
                       'product_item_price',
                       'product_categories',
                       'device_type_user_agent',
                       'device_brand_name_user_agent',
                       'device_operating_system_user_agent',
                       'device_browser_user_agent',
                       'repeat_orders',
                       'net_promoter_score',
                       'hit_of_logged_in_user',
                       'registered_user',
                       'user_gender',
                       'user_age',
                       'visit_during_tv_spot']

    # subset dataframe to select only columns to keep
    df = df[columns_to_keep]

    print('Dropping columns complete.')

    return df



def rename_columns(df):

    print('Starting renaming columns...')

    df.rename(columns={'va_closer_id' : 'marketing_channel'}, inplace=True)
    df.rename(columns={'os' : 'operating_system'}, inplace=True)
    df.rename(columns={'ref_type' : 'referrer_type'}, inplace=True)
    df.rename(columns={'post_search_engine' : 'search_engine'}, inplace=True)
    df.rename(columns={'cart_value_(v50)' : 'cart_value'}, inplace=True)
    df.rename(columns={'int._stand._search_result_clicked_(e16)' : 'standard_search_results_clicked'}, inplace=True)
    df.rename(columns={'active_stand._search_started_(e17)' : 'standard_search_started'}, inplace=True)
    df.rename(columns={'sugg._search_result_clicked_(e18)' : 'suggested_search_results_clicked'}, inplace=True)
    df.rename(columns={'post_cookies' : 'cookies'}, inplace=True)
    df.rename(columns={'post_persistent_cookie' : 'persistent_cookie'}, inplace=True)
    df.rename(columns={'repeat_orders_(e9)' : 'repeat_orders'}, inplace=True)
    df.rename(columns={'net_promoter_score_raw_(v10)_-_user' : 'net_promoter_score'}, inplace=True)
    df.rename(columns={'hit_of_logged_in_user_(e23)' : 'hit_of_logged_in_user'}, inplace=True)
    df.rename(columns={'registered_user_(user)_(v34)' : 'registered_user'}, inplace=True)
    df.rename(columns={'user_gender_(v61)' : 'user_gender'}, inplace=True)
    df.rename(columns={'user_age_(v62)' : 'user_age'}, inplace=True)
    df.rename(columns={'visit_during_tv_spot_(e71)' : 'visit_during_tv_spot'}, inplace=True)

    print('Renaming columns complete')

    return df



def fill_missing_and_faulty_values(df):

    print('Starting filling missing and faulty values...')

    df['cart_value'].fillna(0, inplace=True)
    df['registered_user'] = df['registered_user'].apply(lambda x: 1 if x == 'y' else 0)
    df['cookies'] = df['cookies'].apply(lambda x: 1 if x == 'Y' else 0)
    df['persistent_cookie'] = df['persistent_cookie'].apply(lambda x: 1 if x == 'Y' else 0)

    print('Filling missing and faulty values complete.')

    return df



def cast_data_types(df):

    print('Starting casting data types...')

    # datetime columns
    df['date_time'] = df['date_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['hit_time_gmt'] = pd.to_datetime(df['hit_time_gmt'], unit='s')
    df['visit_start_time_gmt'] = pd.to_datetime(df['visit_start_time_gmt'], unit='s')

    # integer columns
    integer_columns = ['visit_num',
                       'visit_page_num',
                       'purchase_boolean',
                       'product_view_boolean',
                       'checkout_boolean',
                       'cart_addition_boolean',
                       'cart_removal_boolean',
                       'cart_view_boolean',
                       'campaign_view_boolean',
                       'page_view_boolean',
                       'last_purchase_num',
                       'standard_search_results_clicked',
                       'standard_search_started',
                       'suggested_search_results_clicked',
                       'cookies',
                       'persistent_cookie',
                       'search_page_num',
                       'new_visit',
                       'hourly_visitor',
                       'daily_visitor',
                       'weekly_visitor',
                       'monthly_visitor',
                       'quarterly_visitor',
                       'yearly_visitor',
                       'repeat_orders',
                       'hit_of_logged_in_user',
                       'registered_user',
                       'visit_during_tv_spot']

    for column in integer_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: int(float(x)))
        else:
            pass

    # float column
    df['cart_value'] = df['cart_value'].apply(lambda x: float(x))

    print('Casting data types complete.')

    return df



### MAPPING

def connection_type_mapping(df):

    print('Starting connection type mapping...')

    # load file for connection type mapping and select columns
    connection_type_mapping = pd.read_csv('../data/mapping_files/connection_type.tsv', sep='\t', header=None)
    connection_type_mapping.columns = ['connection_type_id', 'connection_type_name']

    # create dictionary for connection type mapping
    connection_type_mapping_dict = dict(zip(connection_type_mapping.connection_type_id, connection_type_mapping.connection_type_name))

    # map connection types
    df['connection_type'] = df['connection_type'].map(connection_type_mapping_dict).fillna(df['connection_type'])

    print('Connection type mapping complete.')

    return df



def country_mapping(df):

    print('Starting country mapping...')

    # load file for country mapping and select columns
    country_mapping = pd.read_csv('../data/mapping_files/country.tsv', sep='\t', header=None)
    country_mapping.columns = ['country_id', 'country_name']

    # drop dupliate countries
    country_mapping = country_mapping.drop_duplicates('country_name').reset_index(drop=True)

    # create dictionary for country mapping
    country_mapping_dict = dict(zip(country_mapping.country_id, country_mapping.country_name))

    # map countries
    df['country'] = df['country'].map(country_mapping_dict).fillna(df['country'])

    print('Country mapping complete.')

    return df



def custom_evars_mapping(df):

    print('Starting custom evars mapping...')

    # load file for custom evars mapping and select columns
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

    return df



def custom_marketing_channel_mapping(df):

    print('Starting custom marketing channel mapping...')

    # load file for custom marketing channel mapping
    custom_marketing_channel_mapping = pd.read_csv('../data/mapping_files/custom_marketing_channels.tsv', sep='\t')
	
    # create dictionary for marketing channel mapping
    custom_marketing_channel_mapping_dict = dict(zip(custom_marketing_channel_mapping.channel_id, custom_marketing_channel_mapping.name))

    # map custom marketing channels
    df['va_closer_id'] = df['va_closer_id'].apply(lambda x: float(x))
    df['va_closer_id'] = df['va_closer_id'].map(custom_marketing_channel_mapping_dict).fillna(df['va_closer_id'])
    df['va_closer_id'] = df['va_closer_id'].apply(lambda x: 'Unknown' if x == 0 else x)

    print('Custom marketing channel mapping complete.')

    return df



def custom_and_standard_events_mapping(df):

    print('Starting custom and standard events mapping...')

    # fill missing values in post_event_list
    df['post_event_list'] = df['post_event_list'].fillna('Unknown')

    # load file for standard event mapping and select columns
    standard_events = pd.read_csv('../data/mapping_files/event.tsv', sep='\t', header=None)
    standard_events.columns = ['event_id', 'event_name']

    # load file for custom event mapping and modify event_id for matching
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
    df = df[df['internal_user_(e30)'] != 1]

    print('Standard and custom events mapping complete.')

    return df



def referrer_type_mapping(df):

    print('Starting referrer type mapping...')

    # load file for referrer type mapping and select columns
    referrer_type_mapping = pd.read_csv('../data/mapping_files/referrer_type.tsv', sep='\t', header=None)
    referrer_type_mapping.columns = ['referrer_type_id', 'referrer_type_name', 'referrer_type']

    # create dictionary for referrer type mapping
    referrer_type_mapping_dict = dict(zip(referrer_type_mapping.referrer_type_id, referrer_type_mapping.referrer_type))

    # map referrer types
    df['ref_type'] = df['ref_type'].map(referrer_type_mapping_dict).fillna(df['ref_type'])

    print('Referrer type mapping complete.')

    return df



def search_engine_mapping(df):

    print('Starting search engine mapping...')

    # load file for search engine mapping and select columns
    search_engine_mapping = pd.read_csv('../data/mapping_files/search_engines.tsv', sep='\t', header=None)
    search_engine_mapping.columns = ['search_engine_id', 'search_engine_name']

    # create dictionary for search engine mapping
    search_engine_mapping_dict = dict(zip(search_engine_mapping.search_engine_id, search_engine_mapping.search_engine_name))

    # map search engines
    df['post_search_engine'] = df['post_search_engine'].map(search_engine_mapping_dict).fillna(df['post_search_engine'])

    # clean search_engine and keep only general search engine name
    df['post_search_engine'] = df['post_search_engine'].apply(lambda x: str(x).split(' ')[0] if pd.notnull(x) else x)
    df['post_search_engine'] = df['post_search_engine'].apply(lambda x: 'Google' if x == 'googleadservices.com' else x)
    df['post_search_engine'] = df['post_search_engine'].apply(lambda x: 'Unknown' if x == '0.0' else x)

    print('Search engine mapping complete.')

    return df



def generate_user_agent_mapping(df):

    print('Starting user agent device type, brand name, operating system and bot flag mapping...')

    # fill missing values
    df['user_agent'] = df['user_agent'].fillna('Unknown')

    # create dataframe for user agent mapping and fill with unique user agents
    columns = ['user_agent',
               'device_type_user_agent',
               'device_brand_name_user_agent',
               'device_operating_system_user_agent',
               'device_browser_user_agent',
               'device_is_bot_user_agent']
    index = np.arange(df['user_agent'].nunique())
    user_agent_mapping_df = pd.DataFrame(index=index, columns=columns)
    user_agent_mapping_df['user_agent'] = df['user_agent'].unique()

    # map device type
    user_agent_mapping_df['device_type_user_agent'] = user_agent_mapping_df['user_agent'].apply(lambda x: DeviceDetector(x).parse().device_type())
    user_agent_mapping_df['device_type_user_agent'] = user_agent_mapping_df['device_type_user_agent'].apply(lambda x: 'Unknown' if x == '' else x)

    # map brand name
    user_agent_mapping_df['device_brand_name_user_agent'] = user_agent_mapping_df['user_agent'].apply(lambda x: DeviceDetector(x).parse().device_brand_name())
    user_agent_mapping_df['device_brand_name_user_agent'] = user_agent_mapping_df['device_brand_name_user_agent'].apply(lambda x: 'Unknown' if x == 'UNK' else x)

    # map operating system
    user_agent_mapping_df['device_operating_system_user_agent'] = user_agent_mapping_df['user_agent'].apply(lambda x: DeviceDetector(x).parse().os_name())
    user_agent_mapping_df['device_operating_system_user_agent'] = user_agent_mapping_df['device_operating_system_user_agent'].apply(lambda x: 'Unknown' if x == '' else x)

    # map browser
    user_agent_mapping_df['device_browser_user_agent'] = user_agent_mapping_df['user_agent'].apply(lambda x: DeviceDetector(x).parse().client_name())

    # map bot flag
    user_agent_mapping_df['device_is_bot_user_agent'] = user_agent_mapping_df['user_agent'].apply(lambda x: DeviceDetector(x).parse().is_bot())
    user_agent_mapping_df['device_is_bot_user_agent'] = user_agent_mapping_df['device_is_bot_user_agent'].apply(lambda x: 1 if x == True else 0)

    print('User agent device type, brand name, operating system, browser and bot flag mapping complete.')

    return user_agent_mapping_df



def user_agent_mapping(df):

    print('Starting user agent mapping...')

    # fill missing values
    df['user_agent'] = df['user_agent'].fillna('Unknown')

    # load file for user agent mapping
    user_agent_mapping = pd.read_pickle('../data/mapping_files/user_agent_mapping.pkl.gz', compression='gzip')

    # merge user agent mapping and df
    df = pd.merge(df, user_agent_mapping, how='left', on='user_agent')

    # drop rows where device_is_bot_user_agent == 1
    df = df.drop(df[df['device_is_bot_user_agent'] == 1].index)

    # fill missing values in user agent columns (device_type_user_agent 51, device_brand_name_user_agent 51, device_operating_system_user_agent 51 and device_browser_user_agent 842 missing rows)
    device_type_columns = [x for x in df.columns if x.lower()[:6] == 'device']
    for i in device_type_columns:
        df[i] = df[i].apply(lambda x: 'Unknown' if pd.isnull(x) else x)

    print('User agent mapping complete.')

    return df



### PROCESSING

def process_product_items(df):

    print('Starting processing product items...')

    df['product_items'] = df['product_items'].apply(lambda x: len([x for x in x.split(';') if x]) if pd.notnull(x) else 0)

    print('Processing product items complete.')

    return df



def process_product_item_prices(df):

    print('Starting processing product item prices...')

    df['product_item_price'] = df['product_item_price'].apply(lambda x: sum([float(x) for x in x.split(';') if x]) if (pd.notnull(x)) & (x != 'product_item_price') else 0)

    print('Processing product item prices complete.')

    return df



def process_product_categories(df):

    print('Starting processing product categories...')
	
    product_categories_level_1 = ['Computer & Elektronik', 
    'Wohnen & Haushalt', 
    'SchÃ¶nheit & Gesundheit', 
    'Baumarkt & Garten',
    'Baby & Spielzeug',
    'Sport & Freizeit',
    'Mode & Accessoires',
    'Lebensmittel & GetrÃ¤nke',
    'Medien & Unterhaltung']
	
    for i in ['_first', '_last']:
	
        df['product_categories'+str(i)] = df['product_categories'+str(i)].apply(lambda x: 'Unknown' if pd.isnull(x) else x.split('/'))
        df['product_categories'+str(i)] = df['product_categories'+str(i)].apply(lambda x: x if x == 'Unknown' else [x.strip() for x in x][0])
	
        df['product_categories_level_1'+str(i)] = df['product_categories'+str(i)].apply(lambda x: x if x in product_categories_level_1 else 'Unknown')

        df.drop('product_categories'+str(i), axis=1, inplace=True)
		
    print('Processing product categories complete.')

    return df

	
	
def process_net_promoter_score(df):

    print('Starting processing net promoter score...')
	
    for i in ['_first', '_last']:
	
        df['net_promoter_score'+str(i)] = df['net_promoter_score'+str(i)].apply(lambda x: 'Unknown' if pd.isnull(x) else ('8' if x == '8th' else str(int(x))))

    print('Processing net promoter score complete.')
	
    return df

	

def process_user_gender(df):

    print('Starting processing user gender...')
	
    for i in ['_first', '_last']:
	
        df['user_gender'+str(i)] = df['user_gender'+str(i)].apply(lambda x: 'Unknown' if pd.isnull(x) else ('female' if x == 'Frau' else 'male'))
 
    print('Processing user gender complete.')
   
    return df
	

	
def process_user_age(df):

    print('Starting user age...')

    for i in ['_first', '_last']:
	
        df['user_age'+str(i)] = df['user_age'+str(i)].apply(lambda x: 0 if pd.isnull(x) else (int(x) if re.match('^([1][9][0-9][0-9]|[2][0][0][0-2])$', x) else 0))

    print('Processing user age complete.')
	
    return df
	

	
def process_search_engines(df):

    print('Starting processing search engines...')

    search_engines_to_keep = ['Google', 
	'Unknown', 
	'Microsoft', 
	'Yahoo!']
	
    for i in ['_first', '_last']:
	
        df['search_engine_reduced'+str(i)] = df['search_engine'+str(i)].apply(lambda x: x if x in search_engines_to_keep else 'Other')

        df.drop('search_engine'+str(i), axis=1, inplace=True)
		
    print('Processing search engines complete.')
	
    return df
	
	
def process_device_types(df):

    print('Starting processing device types...')

    device_types_to_keep = ['smartphone', 
	'desktop', 
	'tablet', 
	'phablet', 
	'Unknown', 
	'portable media player']
	
    for i in ['_first', '_last']:
	
        df['device_type_user_agent_reduced'+str(i)] = df['device_type_user_agent'+str(i)].apply(lambda x: x if x in device_types_to_keep else 'Other')

        df.drop('device_type_user_agent'+str(i), axis=1, inplace=True)		
		
    print('Processing device types complete.')
	
    return df	


	
def process_device_brand_names(df):

    print('Starting processing device brand names...')

    device_brand_names_to_keep = ['Apple', 
	'Unknown', 
	'Samsung', 
	'Sony', 
	'HTC', 
	'Huawei', 
	'Google', 
	'LG', 
	'Nokia', 
	'Microsoft', 
	'Wiko', 
	'Lenovo', 
	'Acer', 
	'Asus', 
	'RIM', 
	'Motorola', 
	'OnePlus', 
	'Toshiba']
	
    for i in ['_first', '_last']:
	
        df['device_brand_name_user_agent_reduced'+str(i)] = df['device_brand_name_user_agent'+str(i)].apply(lambda x: x if x in device_brand_names_to_keep else 'Other')
        
        df.drop('device_brand_name_user_agent'+str(i), axis=1, inplace=True)		

    print('Processing device brand names complete.')
	
    return df
	
	
	
def process_device_operating_systems(df):

    print('Starting processing device operating systems...')

    device_operating_systems_to_keep = ['iOS',
	'Windows',
	'Android',
	'Mac',
	'Windows Phone',
	'GNU/Linux',
	'Unknown',
	'Ubuntu',
	'Chrome OS']
	
    for i in ['_first', '_last']:
	
        df['device_operating_system_user_agent_reduced'+str(i)] = df['device_operating_system_user_agent'+str(i)].apply(lambda x: x if x in device_operating_systems_to_keep else 'Other')    

        df.drop('device_operating_system_user_agent'+str(i), axis=1, inplace=True)
		
    print('Processing device operating systems complete.')

    return df

	
	
def process_device_browsers(df):

    print('Starting processing device browsers...')

    device_browsers_to_keep = ['Mobile Safari',
	'Chrome',
	'Chrome Mobile',
	'Internet Explorer',
	'Samsung Browser',
	'Firefox',
	'Facebook',
	'Safari',
	'Microsoft Edge',
	'Android Browser',
	'Chrome Mobile iOS',
	'Firefox Mobile',
	'IE Mobile',
	'Opera',
	'Blackberry Browser',
	'Opera Mobile']	

    for i in ['_first', '_last']:	
	
        df['device_browser_user_agent_reduced'+str(i)] = df['device_browser_user_agent'+str(i)].apply(lambda x: x if x in device_browsers_to_keep else 'Other')    

        df.drop('device_browser_user_agent'+str(i), axis=1, inplace=True)
		
    print('Processing device browsers complete.')

    return df
	
	
	
### DESCRIPTIVES

def save_run_time(run_time_dict_file, run_time_dict):

    f = open('../results/descriptives/'+run_time_dict_file, 'w')
    f.write(str(run_time_dict))
    f.close()