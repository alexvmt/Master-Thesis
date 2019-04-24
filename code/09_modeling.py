#!/usr/bin/env python
# coding: utf-8

##### MODELING #####

print('Starting modeling...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import os
import sys
params = sys.argv

import numpy as np
import pandas as pd

from helper_functions import *

# print selected training set, test set and model
print('Training set size:', params[1], 'unique visitors')
print('Test set size:', int(int(params[1])/0.8*0.2), 'unique_visitors')
print('Model:', params[2])



### LOAD DATA
print('Starting loading training and test sets...')

# input files
input_file_train = 'train_'+params[1]+'.pkl.gz'
input_file_test = 'test_'+params[1]+'.pkl.gz'

# output files
output_file_model_performance = 'model_performance_'+params[1]+'_'+params[2]+'.pkl.gz'
output_file_coefficients = 'coefficients_'+params[1]+'_'+params[2]+'.pkl.gz'
output_file_feature_importances = 'feature_importances_'+params[1]+'_'+params[2]+'.pkl.gz'

# load training and test sets
train = pd.read_pickle('../data/training_and_test_sets/'+input_file_train)
test = pd.read_pickle('../data/training_and_test_sets/'+input_file_test)

# extract target and features from training and test sets
y_train = train['purchase_within_next_24_hours']
y_test = test['purchase_within_next_24_hours']
unique_visitors_train = train['visitor_id'].nunique()
unique_visitors_test = test['visitor_id'].nunique()
X_train = train.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours'], axis=1)
X_test = test.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours'], axis=1)

print('Loading training and test sets complete.')



### PREPARE MODEL

# set seed
random_state = np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

# import model
if params[2] == 'LR':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=random_state)

elif params[2] == 'DT':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=random_state)

elif params[2] == 'NB':
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()

elif params[2] == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()

elif params[2] == 'RF':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=random_state)

elif params[2] == 'SVM':
    from sklearn.svm import LinearSVC
    model = LinearSVC(random_state=random_state)

elif params[2] == 'BOOST':
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(random_state=random_state)

elif params[2] == 'NN1':
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import EarlyStopping

    # fully connected 1 layer neural network with as many neurons per input and hidden layer as there are features in the training set
    # ReLU activation in all layers except for Sigmoid activation in output layer
    # Xavier uniform initializer for weight initialization
    # 20% dropout after input and all hidden layers to avoid overfitting
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

elif params[2] == 'NN3':
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import EarlyStopping

    # fully connected 3 layer neural network with as many neurons per input and hidden layer as there are features in the training set
    # ReLU activation in all layers except for Sigmoid activation in output layer
    # Xavier uniform initializer for weight initialization
    # 20% dropout after input and all hidden layers to avoid overfitting
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

elif params[2] == 'NN5':
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import EarlyStopping

    # fully connected 5 layer neural network with as many neurons per input and hidden layer as there are features in the training set
    # ReLU activation in all layers except for Sigmoid activation in output layer
    # Xavier uniform initializer for weight initialization
    # 20% dropout after input and all hidden layers to avoid overfitting
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2, seed=random_state))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

elif params[2] in ['RNN', 'LSTM']:
    from collections import defaultdict
    import copy
    from keras.models import Sequential
    from keras.layers import SimpleRNN, LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping
    from math import floor

    print('Transforming 2D training and test dataframes to 3D numpy arrays...')

    # select features
    feature_columns = list(train)
    [feature_columns.remove(column) for column in ['visitor_id', 'visit_start_time_gmt']]

    # remove features that capture behavior in previous visits
    temporal_features_dropped = len([x for x in feature_columns if '_in_last_' in x])
    feature_columns = [x for x in feature_columns if '_in_last_' not in x]

    # select target
    target_column = 'purchase_within_next_24_hours'
    feature_columns.remove(target_column)

    # build 3D numpy arrays for training from 2D training set
    X_train_generator = defaultdict(lambda: defaultdict(list))
    y_train_generator = defaultdict(lambda: defaultdict(lambda: 0))
    visit_counter = defaultdict(lambda: 1)
    for index, row in train.iterrows():
        curr_row = [row[x] for x in feature_columns]
        visitor_id = row['visitor_id']
        curr_visit_count = visit_counter[visitor_id]
        if curr_visit_count > 1:
            X_train_generator[visitor_id][curr_visit_count] = copy.deepcopy(X_train_generator[visitor_id][curr_visit_count - 1])
        X_train_generator[visitor_id][curr_visit_count].append(curr_row)
        y_train_generator[visitor_id][curr_visit_count] = int(row[target_column])
        visit_counter[visitor_id] += 1

        # build 3D numpy arrays for testing from 2D test set
    X_test_generator = defaultdict(lambda: defaultdict(list))
    y_test_generator = defaultdict(lambda: defaultdict(lambda: 0))
    visit_counter = defaultdict(lambda: 1)
    for index, row in test.iterrows():
        curr_row = [row[x] for x in feature_columns]
        visitor_id = row['visitor_id']
        curr_visit_count = visit_counter[visitor_id]
        if curr_visit_count > 1:
            X_test_generator[visitor_id][curr_visit_count] = copy.deepcopy(X_test_generator[visitor_id][curr_visit_count - 1])
        X_test_generator[visitor_id][curr_visit_count].append(curr_row)
        y_test_generator[visitor_id][curr_visit_count] = int(row[target_column])
        visit_counter[visitor_id] += 1

    # build train and test generators from 3D numpy arrays for training and testing
    def data_generator(features, target, batch_size=1):
        for visitor in features:
            for visit in features[visitor]:
                yield np.array([features[visitor][visit]]), np.array([target[visitor][visit]])
    generator_train = data_generator(X_train_generator, y_train_generator)
    generator_test = data_generator(X_test_generator, y_test_generator)

    print('Transforming 2D training and test dataframes to 3D numpy arrays complete.')

    if params[2] == 'RNN':
        # fully conncected recurrent neural network with 1 recurrent layer consistent of 256 units
        # tanh activation in recurrent layer and Sigmoid activation in output layer
        # Xavier uniform initializer for weight initialization
        # 20% dropout in recurrent layer to avoid overfitting
        model = Sequential()
        model.add(SimpleRNN(256, input_shape=(None, len(feature_columns)), return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
    elif params[2] == 'LSTM':
        # recurrent neural network with 1 LSTM layer cosistent of 256 units
        # tanh activation in LSTM layer and Sigmoid activation in output layer
        # Xavier uniform initializer for weight initialization
        # 20% dropout in LSTM layer to avoid overfitting
        model = Sequential()
        model.add(LSTM(256, input_shape=(None, len(feature_columns)), return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        pass

else:
    print('Model not found. Please select one of the following models: LR, DT, NB, KNN, RF, SVM, BOOST, NN1, NN3, NN5, RNN or LSTM.')



### TRAIN
print('Training '+params[2]+'...')
training_start_time = datetime.now()

if params[2] in ['NN1', 'NN3', 'NN5']:
    epochs = 10
    batch_size = 256
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)

elif params[2] in ['RNN', 'LSTM']:
    epochs = 10
    steps_per_epoch = floor(X_train.shape[0]/epochs)
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    model.fit_generator(generator_train, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)

else:
    model.fit(X_train, y_train)

training_time = (datetime.now() - training_start_time)
print('Training '+params[2]+' complete, training time: '+str(training_time)+'.')



### TEST
print('Testing '+params[2]+'...')
testing_start_time = datetime.now()

if params[2] in ['NN1', 'NN3', 'NN5']:
    y_pred = model.predict_classes(X_test)

elif params[2] in ['RNN', 'LSTM']:
    steps = X_test.shape[0]
    y_pred = np.around(model.predict_generator(generator_test, steps=steps)).astype(int)

else:
    y_pred = model.predict(X_test)

testing_time = datetime.now() - testing_start_time
print('Testing '+params[2]+' complete, test time: '+str(testing_time)+'.')



### EVALUATE
print('Evaluating '+params[2]+'...')

# import performance metrics
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report

# set up dataframe for model performance
columns = ['Unique visitors train',
'Unique visitors test',
'Sessions train',
'Sessions test',
'Features',
'Training time',
'Testing time',
'Accuracy',
'AUC',
'True negatives',
'False negatives',
'True positives',
'False positives',
'Precision',
'Recall',
'F-score']
index = [params[2]]
model_performance = pd.DataFrame(index=index, columns=columns)

# calculate performance metrics and save them in dataframe
model_performance['Unique visitors train'] = unique_visitors_train
model_performance['Unique visitors test'] = unique_visitors_test

model_performance['Sessions train'] = X_train.shape[0]
model_performance['Sessions test'] = X_test.shape[0]

if params[2] in ['RNN', 'LSTM']:
    model_performance['Features'] = X_train.shape[1]-temporal_features_dropped
else:
    model_performance['Features'] = X_train.shape[1]
	
model_performance['Training time'] = training_time.seconds
model_performance['Testing time'] = testing_time.seconds

model_performance['Accuracy'] = accuracy_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
model_performance['AUC'] = auc(fpr, tpr)

true_negatives = confusion_matrix(y_test, y_pred)[0,0]
false_negatives = confusion_matrix(y_test, y_pred)[1,0]
true_positives = confusion_matrix(y_test, y_pred)[1,1]
false_positives = confusion_matrix(y_test, y_pred)[0,1]
model_performance['True negatives'] = true_negatives
model_performance['False negatives'] = false_negatives
model_performance['True positives'] = true_positives
model_performance['False positives'] = false_positives

precision = true_positives/(true_positives+false_positives)
recall = true_positives/(true_positives+false_negatives)
f_score = 2*((precision*recall)/(precision+recall))
model_performance['Precision'] = precision
model_performance['Recall'] = recall
model_performance['F-score'] = f_score

# save model performance dataframe
model_performance.to_pickle('../results/model_performance/'+output_file_model_performance, compression='gzip')

# save coefficients if selected model is LR or SVM
if params[2] in ['LR', 'SVM']:
    print('Starting saving coefficients of '+params[2]+'...')
    columns = ['Features']
    index = np.arange(X_train.shape[1])
    coefficients = pd.DataFrame(index=index, columns=columns)
    coefficients['Features'] = X_train.columns
    coefficients['Coefficients'] = model.coef_[0][:]
    coefficients.to_pickle('../results/model_performance/'+output_file_coefficients, compression='gzip')
    print('Saving coefficients of '+params[2]+' complete.')
else:
    pass
	
# save feature importances if selected model is DT, RF or BOOST
if params[2] in ['DT', 'RF', 'BOOST']:
    print('Starting saving feature importances of '+params[2]+'...')
    columns = ['Features']
    index = np.arange(X_train.shape[1])
    feature_importances = pd.DataFrame(index=index, columns=columns)
    feature_importances['Features'] = X_train.columns
    feature_importances['Feature importances'] = model.feature_importances_
    feature_importances.to_pickle('../results/model_performance/'+output_file_feature_importances, compression='gzip')
    print('Saving feature importances of '+params[2]+' complete.')
else:
    pass

print('Evaluating '+params[2]+' complete.')



print('Modeling complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/modeling_run_time_'+params[1]+'_'+params[2]+'.txt', run_time)
