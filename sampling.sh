# OLD

gzip -cd 25_week_sample_raw.tsv.gz >25_week_sample_raw.tsv

grep -P "\t2016-05-([0][9]|[1][0-1]) \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >3_day_sample_raw.tsv
gzip 3_day_sample_raw.tsv

grep -P "\t2016-(([0][5]-\d\d)|([0][6]-[0-1][0-9])) \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >6_week_sample_raw.tsv
gzip 6_week_sample_raw.tsv

grep -P "\t2016-([0][5-7])-\d\d \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >12_week_sample_raw.tsv
gzip 12_week_sample_raw.tsv

grep -P "\t2016-(([0][5-9]-\d\d)|([1][0]-[0-1][0-9])|([1][0]-[2][0-4])) \d\d:\d\d:\d\d\t" 25_week_sample_raw.tsv >24_week_sample_raw.tsv
gzip 24_week_sample_raw.tsv



# NEW

grep -P "\t2016-05-\d\d \d\d:\d\d:\d\d\t" clickstream_0516-1016_raw.tsv >clickstream_0516_raw.tsv
gzip clickstream_0516_raw.tsv

grep -P "\t2016-06-\d\d \d\d:\d\d:\d\d\t" clickstream_0516-1016_raw.tsv >clickstream_0616_raw.tsv
gzip clickstream_0616_raw.tsv

grep -P "\t2016-07-\d\d \d\d:\d\d:\d\d\t" clickstream_0516-1016_raw.tsv >clickstream_0716_raw.tsv
gzip clickstream_0716_raw.tsv

grep -P "\t2016-08-\d\d \d\d:\d\d:\d\d\t" clickstream_0516-1016_raw.tsv >clickstream_0816_raw.tsv
gzip clickstream_0816_raw.tsv

grep -P "\t2016-09-\d\d \d\d:\d\d:\d\d\t" clickstream_0516-1016_raw.tsv >clickstream_0916_raw.tsv
gzip clickstream_0916_raw.tsv

grep -P "\t2016-10-\d\d \d\d:\d\d:\d\d\t" clickstream_0516-1016_raw.tsv >clickstream_1016_raw.tsv
gzip clickstream_1016_raw.tsv