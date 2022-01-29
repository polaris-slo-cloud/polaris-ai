#!/bin/bash

model_path="/home/tuwien/amorichetta/polaris-ai/predictive_monitoring/models/lstm_model_multistepahead"
output_path="/home/tuwien/amorichetta/polaris-ai/predictive_monitoring/experiments_result/figures_LSTM/lstm_multistepahead_test/"
mongodb_name="predictiveDB"
mongodb_collection="multistepahead"
lookback=24
lookahead=3

screen -d -m -S batch-00-test /home/tuwien/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/bash/multistepahead_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/aggregated_multistep_tests/aggregated_list_00_of_05.txt ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead}

screen -d -m -S batch-01-test /home/tuwien/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/bash/multistepahead_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/aggregated_multistep_tests/aggregated_list_01_of_05.txt ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead}

screen -d -m -S batch-02-test /home/tuwien/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/bash/multistepahead_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/aggregated_multistep_tests/aggregated_list_02_of_05.txt ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead}

screen -d -m -S batch-03-test /home/tuwien/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/bash/multistepahead_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/aggregated_multistep_tests/aggregated_list_03_of_05.txt ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead}

screen -d -m -S batch-04-test /home/tuwien/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/bash/multistepahead_single-job_prediction.sh /home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/aggregated_multistep_tests/aggregated_list_04_of_05.txt ${model_path} ${output_path} ${mongodb_name} ${mongodb_collection} ${lookback} ${lookahead}