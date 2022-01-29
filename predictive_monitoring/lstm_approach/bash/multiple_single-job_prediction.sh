#!/bin/bash

#job_id=3418339
job_file=$1
input_path=$2
model_path=$3
output_path=$4
mongodb_name=$5
mongodb_collection=$6

source /home/tuwien/anaconda3/etc/profile.d/conda.sh
conda activate neuralnets
while read jobid; do
    python /home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/gcd_test_model.py ${jobid} ${input_path} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection}
done < $job_file

