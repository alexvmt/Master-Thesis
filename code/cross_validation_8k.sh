#!/usr/bin/env bash

python -W ignore 08_cross_validation.py 6250 LR &
python -W ignore 08_cross_validation.py 6250 DT &
python -W ignore 08_cross_validation.py 6250 NB &
python -W ignore 08_cross_validation.py 6250 KNN &
python -W ignore 08_cross_validation.py 6250 RF &
python -W ignore 08_cross_validation.py 6250 SVM &
python -W ignore 08_cross_validation.py 6250 BOOST &
python -W ignore 08_cross_validation.py 6250 NN1 &
python -W ignore 08_cross_validation.py 6250 NN3 &
python -W ignore 08_cross_validation.py 6250 NN5 &
python -W ignore 08_cross_validation.py 6250 RNN &
python -W ignore 08_cross_validation.py 6250 LSTM &
