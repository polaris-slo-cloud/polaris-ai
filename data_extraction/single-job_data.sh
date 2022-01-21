#!/usr/bin/bash

job_id=$1
input_path=$2
tmp_output_path=$3
output_path=$4

for minval in {00..450..50}; do 
        python single-job_data_extraction.py ${job_id} ${minval} ${input_path} ${tmp_output_path}
done

python create_single-job_dataframe.py ${job_id} ${input_path} ${tmp_output_path} ${output_path}