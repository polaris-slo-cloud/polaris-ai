#!/bin/bash

job_file=$1
model_path=$2
output_path=$3
mongodb_name=$4
mongodb_collection=$5
lookback=$6
lookahead=$7
gpu_id=$8

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate neuralnets
while read jobid; do
    start=`date +%s`
    python /home/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/multistepahead_test.py ${jobid} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead} ${gpu_id}
    end=`date +%s`
    runtime=$((end-start))
    echo "$jobid $runtime" >> /tmp/runtime.log
done < $job_file

