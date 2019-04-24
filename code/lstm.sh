#!/usr/bin/env bash

python -W ignore 09_modeling.py 3125 LSTM
python -W ignore 09_modeling.py 6250 LSTM
python -W ignore 09_modeling.py 12500 LSTM
python -W ignore 09_modeling.py 25000 LSTM
python -W ignore 09_modeling.py 50000 LSTM
python -W ignore 09_modeling.py 100000 LSTM
python -W ignore 09_modeling.py 200000 LSTM
python -W ignore 09_modeling.py 400000 LSTM
python -W ignore 09_modeling.py 800000 LSTM
python -W ignore 09_modeling.py 1600000 LSTM
