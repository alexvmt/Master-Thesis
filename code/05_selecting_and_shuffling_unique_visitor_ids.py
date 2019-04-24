#!/usr/bin/env python
# coding: utf-8

##### SELECTING AND SHUFFLING UNIQUE VISITOR IDS #####

print('Starting selecting and shuffling unique visitor ids...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import numpy as np
import pandas as pd
from random import shuffle
import pickle

from helper_functions import *



### LOAD DATA
print('Starting loading data...')

# input file
input_file = 'clickstream_0516-1016_prepared.pkl.gz'

# output file
output_file = 'unique_visitor_ids.pkl'

# load data
df = pd.read_pickle('../data/processed_data/'+input_file)

print('Loading data complete.')



### SELECT AND SHUFFLE UNIQUE VISITOR IDS
print('Starting selecting and shuffling unique visitor ids...')

# shuffle list with unique visitor ids
unique_visitor_ids = list(df['visitor_id'].unique())
shuffle(unique_visitor_ids)

print('Selecting and shuffling unique visitor ids complete.')



### WRITE DATA
print('Starting writing data...')

with open('../data/processed_data/'+output_file, 'wb') as f:
   pickle.dump(unique_visitor_ids, f)

print('Writing data complete.')



print('Selecting and shuffling unique visitor ids complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/selecting_and_shuffling_unique_visitor_ids_run_time.txt', run_time)
