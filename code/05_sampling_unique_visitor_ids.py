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

from helper_functions import *



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

with open('../data/processed_data/'+output_file, 'wb') as f:
   pickle.dump(unique_visitor_ids, f)

print('Writing data complete.')



print('Sampling unique visitor ids complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save run time
save_descriptives('../results/descriptives/sampling_unique_visitor_ids_run_time.txt', run_time)