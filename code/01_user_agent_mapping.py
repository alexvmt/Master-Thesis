
# coding: utf-8

# In[1]:


########## USER AGENT MAPPING ##########


# In[2]:


print('Starting user agent mapping...')


# In[3]:


### import libraries
import pandas as pd
import numpy as np
from datetime import datetime,date
from device_detector import DeviceDetector

start_time = datetime.now()
print('Start time: ', start_time)


# In[4]:


#### SELECT INPUT AND OUTPUT FILES


# In[5]:


input_file = '3_day_sample_raw.tsv.gz'
output_file = '3_day_sample_user_agent_mapping.tsv.gz'
#input_file = '6_week_sample_raw.tsv.gz'
#output_file = '6_week_sample_user_agent_mapping.tsv.gz'
#input_file = '12_week_sample_raw.tsv.gz'
#output_file = '12_week_sample_user_agent_mapping.tsv.gz'
#input_file = '25_week_sample_raw.tsv.gz'
#output_file = '25_week_sample_user_agent_mapping.tsv.gz'

print('Input file selected: ', input_file)
print('Output file selected: ', output_file)


# In[6]:


##### LOAD DATA
print('Loading data...')


# In[7]:


column_headers = pd.read_csv('../data/mapping_files/column_headers.tsv', sep='\t')
df = pd.read_csv('../data/raw_data/'+input_file, compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, names=column_headers)

print('Loading data complete.')


# In[8]:


##### USER AGENT MAPPING
print('Starting user agent mapping...')


# In[9]:


# fill missing values
df['user_agent'] = df['user_agent'].fillna('Unknown')

# create dataframe for user agent mapping
columns = ['user_agent', 
           'device_type_user_agent', 
           'device_brand_name_user_agent', 
           'device_operating_system_user_agent', 
           'device_browser_user_agent', 
           'device_is_bot_user_agent']
index = np.arange(df['user_agent'].nunique())
user_agent_mapping_df = pd.DataFrame(index=index, columns=columns)
user_agent_mapping_df['user_agent'] = df['user_agent'].unique()


# In[10]:


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

# map bot
user_agent_mapping_df['device_is_bot_user_agent'] = user_agent_mapping_df['user_agent'].apply(lambda x: DeviceDetector(x).parse().is_bot())
user_agent_mapping_df['device_is_bot_user_agent'] = user_agent_mapping_df['device_is_bot_user_agent'].apply(lambda x: 1 if x == True else 0)

print('User agent device type, brand name, operating system, browser and is bot mapping complete.')


# In[11]:


##### WRITE DATAFRAME TO FILE
print('Writing dataframe to file...')


# In[12]:


user_agent_mapping_df.to_csv('../data/mapping_files/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)


# In[13]:


print('User agent mapping complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = '3_day_sample_user_agent_mapping_run_time.txt'
#run_time_dict_file = '6_week_sample_user_agent_mapping_run_time.txt'
#run_time_dict_file = '12_week_sample_user_agent_mapping_run_time.txt'
#run_time_dict_file = '25_week_sample_user_agent_mapping_run_time.txt'

run_time_dict = {'user agent mapping run time' : run_time}

f = open('../data/descriptives/'+run_time_dict_file, 'w')
f.write(str(run_time_dict))
f.close()

