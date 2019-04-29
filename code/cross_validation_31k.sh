#!/usr/bin/env bash

python -W ignore 08_cross_validation.py 25000 LR &
python -W ignore 08_cross_validation.py 25000 DT &
python -W ignore 08_cross_validation.py 25000 NB &
python -W ignore 08_cross_validation.py 25000 KNN &
python -W ignore 08_cross_validation.py 25000 RF &
python -W ignore 08_cross_validation.py 25000 SVM &
python -W ignore 08_cross_validation.py 25000 BOOST &
python -W ignore 08_cross_validation.py 25000 NN1 &
python -W ignore 08_cross_validation.py 25000 NN3 &
python -W ignore 08_cross_validation.py 25000 NN5 &
python -W ignore 08_cross_validation.py 25000 RNN &
python -W ignore 08_cross_validation.py 25000 LSTM &
