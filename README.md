# Master Thesis

## Overview

- Topic: Evaluation of Machine Learning and Deep Learning Models for Customer Journey Prediction in E-Commerce - An Executive Perspective
- Author: Alexander Vladimir Merdian-Tarko
- Study program: M. Sc. Business Administration, Media and Technology Management
- Supervisors: Univ.-Prof. Dr. Jörn Grahl and Dr. Matthias Böck (FELD M)
- Department: Department of Digital Transformation and Value Creation
- Faculty: Faculty of Management, Economics and Social Sciences
- University: University of Cologne
- Date: May 02, 2019

## Contents

1. Introduction
2. Methodology
3. Related Work  
3.1 General Comparative Studies
3.2 Comparative Studies Focused on Customer Journey Prediction
4. Model Evaluation Framework
5. Experiments  
5.1 Experimental Setup
5.2 Data
5.3 Descriptive Statistics
5.4 Models
6. Evaluation of Models and Experimental Results  
6.1 Objectivity
6.2 Predictive Accuracy
6.3 Robustness
6.4 Interpretability
6.5 Versatility
6.6 Algorithmic Efficiency
7. Discussion and Managerial Implications
8. Conclusion  
Appendix
References

## Directories overview
- code: contains Python and Shell scripts used for cleaning and processing the data, calculating descriptives and running the models
- results: contains descriptives, cross-validation and model performance metrics
- thesis: contains a Jupyter Notebook with supplementary material and the LaTeX project used to create the final PDF document

## Directories details

### code
Python scripts for the different steps in processing and modeling the data:
- 01_generate_user_agent_mapping.py
- 02_cleaning_and_mapping.py
- 03_aggregation.py
- 04_preparing_target_and_features.py
- 05_selecting_and_shuffling_unique_visitor_ids.py
- 06_feature_selection.py
- 07_preprocessing_and_sampling.py
- 08_cross_validation.py
- 09_modeling.py

Python script containing helper functions used in the other Python scripts:
- helper_functions.py

Python script used to calculate different descriptive statistics:
- stage_and_sample_descriptives.py

Shell script that executes the respective Python scripts to clean and map the data:
- cleaning_and_mapping.sh

Shell script that executes the respective Python scripts to preprocess the data and create samples of different size:
- preprocessing_and_sampling.sh

Shell script that executes the respective Python scripts for cross-validation using samples of different size:
- cross_validation_8k.sh
- cross_validation_31k.sh
- cross_validation_125k.sh
- cross_validation_500k.sh

Shell scripts that execute the respective Python scripts for each models using samples of different size:
- lr.sh
- dt.sh
- nb.sh
- knn.sh
- rf.sh
- svm.sh
- boost.sh
- nn1.sh
- nn3.sh
- nn5.sh
- rnn.sh
- lstm.sh

Shell script that executes the respective Shell scripts for all models above:
- modeling.sh

### results
- cross_validation: contains the model performance of each run of a model and the resulting cross-validation metrics of each model for 4 samples of different size, indicated by the number in the file name
- descriptives: contains different descriptive statistics of the different stages of processing and modeling the data, including script run times
- model_performance: contains predictive performance metrics of each model and sample and also coefficients and feature importances of models where their calculation has been possible

### thesis
- contains graphics and tables used or created in the corresponding Jupyter Notebook and the LaTeX project used to create the final PDF of this Master Thesis

*Note: The data used in this thesis is not included in this repository since parts of it are confidential.*
