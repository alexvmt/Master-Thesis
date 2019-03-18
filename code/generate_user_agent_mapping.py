#!/usr/bin/env python
# coding: utf-8

##### GENERATE USER AGENT MAPPING #####

print('Starting generating user agent mapping...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd
from device_detector import DeviceDetector

from helper_functions import *



### LOAD DATA

print('Loading data...')

# input file
input_file = 'clickstream_0516-1016_raw.tsv.gz'

# output file
output_file = 'user_agent_mapping.tsv.gz'
    
# load column headers
column_headers = pd.read_csv('../data/mapping_files/column_headers.tsv', sep='\t')

# select columns
columns = ['user_agent']

# load data
df = pd.read_csv('../data/raw_data/'+input_file, compression='gzip', sep='\t', encoding='iso-8859-1', quoting=3, low_memory=False, names=column_headers, usecols=columns)

print('Loading data complete...')



### GENERATE USER AGENT MAPPING

user_agent_mapping_df = generate_user_agent_mapping(df)



### WRITE DATA

print('Starting writing data...')

user_agent_mapping_df.to_csv('../data/mapping_files/'+output_file, compression='gzip', sep='\t', encoding='iso-8859-1', index=False)

print('Writing data complete.')



# save run time
print('Generating user agent mapping complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'generate_user_agent_mapping_run_time.txt'
run_time_dict = {'generate user agent mapping run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)