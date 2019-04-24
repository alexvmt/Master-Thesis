#!/usr/bin/env bash

python -W ignore 07_preprocessing_and_sampling.py 3125 &
python -W ignore 07_preprocessing_and_sampling.py 6250 &
python -W ignore 07_preprocessing_and_sampling.py 12500 &
python -W ignore 07_preprocessing_and_sampling.py 25000 &
python -W ignore 07_preprocessing_and_sampling.py 50000 &
python -W ignore 07_preprocessing_and_sampling.py 100000 &
python -W ignore 07_preprocessing_and_sampling.py 200000 &
python -W ignore 07_preprocessing_and_sampling.py 400000 &
python -W ignore 07_preprocessing_and_sampling.py 800000 &
python -W ignore 07_preprocessing_and_sampling.py 1600000 &
