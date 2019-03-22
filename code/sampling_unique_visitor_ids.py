#!/usr/bin/env python
# coding: utf-8

##### SAMPLING UNIQUE VISITOR IDS #####

print('Starting sampling unique visitor ids...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd
from random import shuffle
import pickle



# input file
input_file = 'clickstream_0516-1016_processed_final.pkl.gz'

# output file
output_file = 'unique_visitor_ids.pkl'

# load data
print('Starting loading data...')

df = pd.read_pickle('../data/processed_data/'+input_file, compression='gzip')

print('Loading data complete.')



# create list with unique visitor ids and shuffle list
unique_visitor_ids = list(df['visitor_id'].unique())
shuffle(unique_visitor_ids)



# write data
print('Starting writing data...')

with open(output_file, 'wb') as f:
   pickle.dump(unique_visitor_ids, f)

print('Writing data complete.')



# save run time
print('Sampling unique visitor ids complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

run_time_dict_file = 'sampling_unique_visitor_ids_run_time.txt'
run_time_dict = {'sampling unique visitor ids run time' : run_time}

save_run_time(run_time_dict_file, run_time_dict)