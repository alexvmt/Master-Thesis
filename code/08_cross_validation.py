#!/usr/bin/env python
# coding: utf-8

##### CROSS VALIDATION #####

print('Starting cross validation...')

# import libraries
from datetime import datetime,date
start_time = datetime.now()
print('Start time: ', start_time)

import os
import sys
params = sys.argv

import numpy as np
import pandas as pd
from random import shuffle

from helper_functions import *

# print parameters passed to script
print('Training set:', params[1])
print('Test set:', 'test'+params[1][5:])
print('Target:', params[2])
print('Model:', params[3])



### LOAD DATA
print('Starting loading training and test sets...')

# verify that correct training set is selected
training_sets = [file for file in os.listdir('../data/training_and_test_sets/') if 'train' in file]
if params[1] in training_sets:
    pass
else:
    print('Training set not found. Please select one of the following training sets: ', training_sets)

# input files
input_file_train = params[1]
input_file_test = 'test'+params[1][5:]

# output files
output_file_cross_validation_metrics = 'cross_validation_metrics_'+params[3]+'_'+params[1][6:-7]+'_'+params[2]+'.pkl.gz'
output_file_cross_validation_runs = 'cross_validation_runs_'+params[3]+'_'+params[1][6:-7]+'_'+params[2]+'.pkl.gz'

# load training and test sets
train = pd.read_pickle('../data/training_and_test_sets/'+input_file_train)
test = pd.read_pickle('../data/training_and_test_sets/'+input_file_test)

# verify that correct target is selected
targets = ['purchase_within_next_24_hours', 'purchase_within_next_7_days']
if params[2] in targets:
    pass
else:
    print('Target not found. Please select one of the following targets: ', targets)

print('Loading training and test sets complete.')



### PREPARE CROSS VALIDATION
print('Starting preparing cross validation...')

# concatenate training and test sets
train_test = train.append(test, ignore_index=True)

# extract unique visitor ids and shuffle them
unique_visitor_ids = list(train_test['visitor_id'].unique())
shuffle(unique_visitor_ids)

# create folds and respective training and test sets
folds = [unique_visitor_ids[i:i + int(len(unique_visitor_ids)/5)] for i in range(0, len(unique_visitor_ids), int(len(unique_visitor_ids)/5))]

train1 = train_test[(train_test['visitor_id'].isin(folds[0])) | train_test['visitor_id'].isin(folds[1]) | train_test['visitor_id'].isin(folds[2]) | train_test['visitor_id'].isin(folds[3])]
test1 = train_test[train_test['visitor_id'].isin(folds[4])]

train2 = train_test[(train_test['visitor_id'].isin(folds[0])) | train_test['visitor_id'].isin(folds[1]) | train_test['visitor_id'].isin(folds[2]) | train_test['visitor_id'].isin(folds[4])]
test2 = train_test[train_test['visitor_id'].isin(folds[3])]

train3 = train_test[(train_test['visitor_id'].isin(folds[0])) | train_test['visitor_id'].isin(folds[1]) | train_test['visitor_id'].isin(folds[4]) | train_test['visitor_id'].isin(folds[3])]
test3 = train_test[train_test['visitor_id'].isin(folds[2])]

train4 = train_test[(train_test['visitor_id'].isin(folds[0])) | train_test['visitor_id'].isin(folds[4]) | train_test['visitor_id'].isin(folds[2]) | train_test['visitor_id'].isin(folds[3])]
test4 = train_test[train_test['visitor_id'].isin(folds[1])]

train5 = train_test[(train_test['visitor_id'].isin(folds[4])) | train_test['visitor_id'].isin(folds[1]) | train_test['visitor_id'].isin(folds[2]) | train_test['visitor_id'].isin(folds[3])]
test5 = train_test[train_test['visitor_id'].isin(folds[0])]

# prepare cross validation loop
training_sets = [train1, train2, train3, train4, train5]
test_sets = [test1, test2, test3, test4, test5]
runs = ['run1', 'run2', 'run3', 'run4', 'run5']

# import performance metrics
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report

# set up dataframe for model performance
columns = ['training_set_size',
'test_set_size',
'target',
'features',
'training_time',
'testing_time',
'accuracy',
'auc',
'true_negatives',
'false_negatives',
'true_positives',
'false_positives',
'precision',
'recall',
'f_score']
index = [runs]
model_performance = pd.DataFrame(index=index, columns=columns)

print('Preparing cross validation complete...')



### CROSS VALIDATING MODEL
print('Starting cross validating '+params[3]+'...')

for run, train, test in zip(runs, training_sets, test_sets):
    
    print('Starting '+run+'...')

    y_train = train[params[2]]
    y_test = test[params[2]]
    X_train = train.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1)
    X_test = test.drop(['visitor_id', 'visit_start_time_gmt', 'purchase_within_next_24_hours', 'purchase_within_next_7_days'], axis=1)

	

    ### PREPARE MODEL

    # set seed
    random_state = np.random.seed(42)
    from tensorflow import set_random_seed
    set_random_seed(42)

    # import model
    if params[3] == 'LR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=random_state)

    elif params[3] == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=random_state)

    elif params[3] == 'NB':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    elif params[3] == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()

    elif params[3] == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=random_state)

    elif params[3] == 'SVM':
        from sklearn.svm import SVC
        model = SVC(random_state=random_state)

    elif params[3] == 'BOOST':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=random_state)

    elif params[3] == 'BAG':
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(random_state=random_state)

    elif params[3] == 'NN1':
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

    elif params[3] == 'NN3':
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

    elif params[3] == 'NN5':
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

    elif params[3] in ['RNN', 'LSTM']:
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
        if params[2] == 'purchase_within_next_24_hours':
            target_column = 'purchase_within_next_24_hours'
            feature_columns.remove(target_column)
            feature_columns.remove('purchase_within_next_7_days')
        elif params[2] == 'purchase_within_next_7_days':
            target_column = 'purchase_within_next_7_days'
            feature_columns.remove(target_column)
            feature_columns.remove('purchase_within_next_24_hours')
        else:
            pass

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

        if params[3] == 'RNN':
            # fully conncected recurrent neural network with 1 recurrent layer consistent of 256 units
            # tanh activation in recurrent layer and Sigmoid activation in output layer
            # Xavier uniform initializer for weight initialization
            # 20% dropout in recurrent layer to avoid overfitting
            model = Sequential()
            model.add(SimpleRNN(256, input_shape=(None, len(feature_columns)), return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam')
        elif params[3] == 'LSTM':
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
        print('Model not found. Please select one of the following models: LR, DT, NB, KNN, RF, SVM, BOOST, BAG, NN1, NN3, NN5, RNN or LSTM.')



    ### TRAIN
    print('Training '+params[3]+'...')
    training_start_time = datetime.now()

    if params[3] in ['NN1', 'NN3', 'NN5']:
        epochs = 10
        batch_size = 256
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)

    elif params[3] in ['RNN', 'LSTM']:
        epochs = 10
        steps_per_epoch = floor(X_train.shape[0]/epochs)
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
        model.fit_generator(generator_train, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=2)

    else:
        model.fit(X_train, y_train)

    training_time = (datetime.now() - training_start_time)
    print('Training '+params[3]+' complete, training time: '+str(training_time)+'.')



    ### TEST
    print('Testing '+params[3]+'...')
    testing_start_time = datetime.now()

    if params[3] in ['NN1', 'NN3', 'NN5']:
        y_pred = model.predict_classes(X_test)

    elif params[3] in ['RNN', 'LSTM']:
        steps = X_test.shape[0]
        y_pred = np.around(model.predict_generator(generator_test, steps=steps)).astype(int)

    else:
        y_pred = model.predict(X_test)

    testing_time = datetime.now() - testing_start_time
    print('Testing '+params[3]+' complete, test time: '+str(testing_time)+'.')



    ### EVALUATE
    print('Evaluating '+params[3]+'...')

    # calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    true_negatives = confusion_matrix(y_test, y_pred)[0,0]
    false_negatives = confusion_matrix(y_test, y_pred)[1,0]
    true_positives = confusion_matrix(y_test, y_pred)[1,1]
    false_positives = confusion_matrix(y_test, y_pred)[0,1]
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    f_score = 2*((precision*recall)/(precision+recall))

    # add performance metrics to model performance dataframe
    model_performance.at[run, 'training_set_size'] = X_train.shape[0]
    model_performance.at[run, 'test_set_size'] = X_test.shape[0]
    model_performance.at[run, 'target'] = params[2]
    if params[3] in ['RNN', 'LSTM']:
        model_performance.at[run, 'features'] = X_train.shape[1]-temporal_features_dropped
    else:
        model_performance.at[run, 'features'] = X_train.shape[1]
    model_performance.at[run, 'training_time'] = training_time.seconds
    model_performance.at[run, 'testing_time'] = testing_time.seconds
    model_performance.at[run, 'accuracy'] = accuracy
    model_performance.at[run, 'auc'] = auc_score
    model_performance.at[run, 'true_negatives'] = true_negatives
    model_performance.at[run, 'false_negatives'] = false_negatives
    model_performance.at[run, 'true_positives'] = true_positives
    model_performance.at[run, 'false_positives'] = false_positives
    model_performance.at[run, 'precision'] = precision
    model_performance.at[run, 'recall'] = recall
    model_performance.at[run, 'f_score'] = f_score

    print('Evaluating '+params[3]+' complete.')

    print(run+' complete.')

	

# set up dataframe for cross validation metrics
columns = ['training_set_size_mean',
'training_set_size_std',
'test_set_size_mean',
'test_set_size_std',
'target',
'features',
'training_time_mean',
'training_time_std',
'testing_time_mean',
'testing_time_std',
'accuracy_mean',
'accuracy_std',
'auc_mean',
'auc_std',
'true_negatives_mean',
'true_negatives_std',
'false_negatives_mean',
'false_negatives_std',
'true_positives_mean',
'true_positives_std',
'false_positives_mean',
'false_positives_std',
'precision_mean',
'precision_std',
'recall_mean',
'recall_std',
'f_score_mean',
'f_score_std']
index = [params[3]]
cross_validation_metrics = pd.DataFrame(index=index, columns=columns)

# calculate cross validation metrics
cross_validation_metrics['training_set_size_mean'] = model_performance['training_set_size'].mean()
cross_validation_metrics['training_set_size_std'] = model_performance['training_set_size'].std()
cross_validation_metrics['test_set_size_mean'] = model_performance['test_set_size'].mean()
cross_validation_metrics['test_set_size_std'] = model_performance['test_set_size'].std()
cross_validation_metrics['training_time_mean'] = model_performance['training_time'].mean()
cross_validation_metrics['training_time_std'] = model_performance['training_time'].std()
cross_validation_metrics['testing_time_mean'] = model_performance['testing_time'].mean()
cross_validation_metrics['testing_time_std'] = model_performance['testing_time'].std()
cross_validation_metrics['target'] = model_performance['target'][0]
cross_validation_metrics['features'] = model_performance['features'][0]
cross_validation_metrics['accuracy_mean'] = model_performance['accuracy'].mean()
cross_validation_metrics['accuracy_std'] = model_performance['accuracy'].std()
cross_validation_metrics['auc_mean'] = model_performance['auc'].mean()
cross_validation_metrics['auc_std'] = model_performance['auc'].std()
cross_validation_metrics['true_negatives_mean'] = model_performance['true_negatives'].mean()
cross_validation_metrics['true_negatives_std'] = model_performance['true_negatives'].std()
cross_validation_metrics['false_negatives_mean'] = model_performance['false_negatives'].mean()
cross_validation_metrics['false_negatives_std'] = model_performance['false_negatives'].std()
cross_validation_metrics['true_positives_mean'] = model_performance['true_positives'].mean()
cross_validation_metrics['true_positives_std'] = model_performance['true_positives'].std()
cross_validation_metrics['false_positives_mean'] = model_performance['false_positives'].mean()
cross_validation_metrics['false_positives_std'] = model_performance['false_positives'].std()
cross_validation_metrics['precision_mean'] = model_performance['precision'].mean()
cross_validation_metrics['precision_std'] = model_performance['precision'].std()
cross_validation_metrics['recall_mean'] = model_performance['recall'].mean()
cross_validation_metrics['recall_std'] = model_performance['recall'].std()
cross_validation_metrics['f_score_mean'] = model_performance['f_score'].mean()
cross_validation_metrics['f_score_std'] = model_performance['f_score'].std()

# save model performance and cross validation metrics dataframes
model_performance.to_pickle('../results/models/'+output_file_cross_validation_runs, compression='gzip')
cross_validation_metrics.to_pickle('../results/models/'+output_file_cross_validation_metrics, compression='gzip')

print('Cross validating '+params[3]+' complete.')



print('Cross validation complete.')
run_time = datetime.now() - start_time
print('Run time: ', run_time)

# save script run time
save_script_run_time('../results/descriptives/cross_validation_run_time_'+params[3]+'_'+params[1][6:-7]+'_'+params[2]+'.txt', run_time)