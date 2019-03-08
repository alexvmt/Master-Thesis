gzip -cd 25_week_sample_raw.tsv.gz >25_week_sample_raw.tsv

grep -P "\t2016-05-([0][9]|[1][0-1]) \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >3_day_sample_raw.tsv
gzip 3_day_sample_raw.tsv

grep -P "\t2016-(([0][5]-\d\d)|([0][6]-[0-1][0-9])) \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >6_week_sample_raw.tsv
gzip 6_week_sample_raw.tsv

grep -P "\t2016-([0][5-7])-\d\d \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >12_week_sample_raw.tsv
gzip 12_week_sample_raw.tsv

grep -P "\t2016-(([0][5-9]-\d\d)|([1][0]-[0-1][0-9])|([1][0]-[2][0-4])) \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >24_week_sample_raw.tsv
gzip 24_week_sample_raw.tsv