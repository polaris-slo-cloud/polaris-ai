#!/bin/bash

input_path="/home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/scheduling_class_3_29days/"
model_path="/home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/models/lstm_batch72_neurons50_epochs400_do0"
output_path="/home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/figures"
mongodb_name="predictiveDB"
mongodb_collection="highlevel_monitoring"


screen -d -m -S batch-00-test /home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/bash/multiple_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/tmp/electable_jobs_00-of-04.txt ${input_path} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection}

screen -d -m -S batch-01-test /home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/bash/multiple_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/tmp/electable_jobs_01-of-04.txt ${input_path} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection}
#
screen -d -m -S batch-02-test /home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/bash/multiple_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/tmp/electable_jobs_02-of-04.txt ${input_path} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection}
#
screen -d -m -S batch-03-test /home/tuwien/amorichetta/SLOC/predictive_monitoring/high-level_monitoring/bash/multiple_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/tmp/electable_jobs_03-of-04.txt ${input_path} ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection}