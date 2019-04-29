#!/usr/bin/env bash

python -W ignore 02_cleaning_and_mapping.py clickstream_0516_raw.tsv.gz &
python -W ignore 02_cleaning_and_mapping.py clickstream_0616_raw.tsv.gz &
python -W ignore 02_cleaning_and_mapping.py clickstream_0716_raw.tsv.gz &
python -W ignore 02_cleaning_and_mapping.py clickstream_0816_raw.tsv.gz &
python -W ignore 02_cleaning_and_mapping.py clickstream_0916_raw.tsv.gz &
python -W ignore 02_cleaning_and_mapping.py clickstream_1016_raw.tsv.gz &
