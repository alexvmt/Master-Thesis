#!/usr/bin/env bash

python -W ignore 08_cross_validation.py 400000 LR &
python -W ignore 08_cross_validation.py 400000 DT &
python -W ignore 08_cross_validation.py 400000 NB &
python -W ignore 08_cross_validation.py 400000 KNN &
python -W ignore 08_cross_validation.py 400000 RF &
python -W ignore 08_cross_validation.py 400000 SVM &
python -W ignore 08_cross_validation.py 400000 BOOST &
python -W ignore 08_cross_validation.py 400000 NN1 &
python -W ignore 08_cross_validation.py 400000 NN3 &
python -W ignore 08_cross_validation.py 400000 NN5 &
python -W ignore 08_cross_validation.py 400000 RNN &
python -W ignore 08_cross_validation.py 400000 LSTM &
