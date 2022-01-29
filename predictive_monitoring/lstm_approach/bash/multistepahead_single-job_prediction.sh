#!/bin/bash

#job_id=3418339
job_file=$1
model_path=$2
output_path=$3
mongodb_name=$4
mongodb_collection=$5
lookback=$6
lookahead=$7

source /home/tuwien/anaconda3/etc/profile.d/conda.sh
conda activate neuralnets
while read jobid; do
    python /home/tuwien/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/multistepahead_test.py ${jobid} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead}
done < $job_file

