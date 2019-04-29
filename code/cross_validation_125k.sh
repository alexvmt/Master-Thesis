#!/usr/bin/env bash

python -W ignore 08_cross_validation.py 10000 LR &
python -W ignore 08_cross_validation.py 10000 DT &
python -W ignore 08_cross_validation.py 10000 NB &
python -W ignore 08_cross_validation.py 10000 KNN &
python -W ignore 08_cross_validation.py 10000 RF &
python -W ignore 08_cross_validation.py 10000 SVM &
python -W ignore 08_cross_validation.py 10000 BOOST &
python -W ignore 08_cross_validation.py 10000 NN1 &
python -W ignore 08_cross_validation.py 10000 NN3 &
python -W ignore 08_cross_validation.py 10000 NN5 &
python -W ignore 08_cross_validation.py 10000 RNN &
python -W ignore 08_cross_validation.py 10000 LSTM &
