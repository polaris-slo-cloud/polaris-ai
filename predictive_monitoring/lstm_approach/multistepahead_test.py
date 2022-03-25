import sys
from math import sqrt
import numpy as np
from numpy import array

from scipy.signal import savgol_filter

from sklearn.metrics import mean_squared_error

import pandas as pd

from matplotlib import pyplot

import tensorflow as tf

from pymongo import MongoClient

from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers import Dense

from keras.models import Sequential
from keras.models import load_model

from gcd_data_manipulation import data_aggregation
from gcd_data_manipulation import load_data
from gcd_data_manipulation import scale_values
from gcd_data_manipulation import series_to_supervised
from gcd_data_manipulation import reshape_data

from multistepahead_model_evaluation import evaluate_forecast
from multistepahead_model_evaluation import walk_forward_validation

import os

import time
import logging

log_name = '/home/amorichetta/polaris-ai/predictive_monitoring/lstm_approach/logging/multistepahead-test.log'
logging.basicConfig(filename=log_name, encoding='utf-8', level = logging.DEBUG)

script_start = time.time()

job_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]
mongodb_name = sys.argv[4] #default predictiveDB
mongodb_collection = sys.argv[5] #default multistepahead
n_input = int(sys.argv[6])
n_out = int(sys.argv[7])
gpu_id = sys.argv[8]

jobname = os.path.basename(job_path).split('.')[0]
exp_name = "LSTM-multivariate-test" #.format(jobname=jobname)

print(f"Computing {exp_name} for {jobname}")

# instantiate mongoDB client
client = MongoClient()
db = client[mongodb_name]
collection = db[mongodb_collection]

columns_to_consider = [
    "CPU rate",
    "canonical memory usage",
    "assigned memory usage",
    "unmapped page cache",
    "total page cache",
    "maximum memory usage",
    "disk I/O time",
    "local disk space usage",
    "maximum CPU rate",
    "maximum disk IO time",
    "cycles per instruction",
    "memory accesses per instruction",
    "Efficiency",  # target metric
]


def config_gpu(device):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device


def interpolate_df(df):
    df.reset_index(inplace=True)
    df["datetime"] = timestamps_readings = [
        pd.Timestamp(int(t / 1000000) + 1304233200, tz="US/Pacific", unit="s")
        for t in df["end time"].values
    ]
    df.set_index("datetime", inplace=True)
    df.interpolate(method="time", limit_direction="both", inplace=True)

    df.reset_index(inplace=True)
    df.drop(['datetime'], axis=1, inplace=True)
    df.set_index("end time", inplace=True)
    
    return df

    
def plot_results(real_values, predictions, output_path, jobname, last_n=700):
    for i in range(3):
        step_y = [x[0] for x in real_values]
        step_yhat = [x[0] for x in predictions]

        pyplot.clf()
        pyplot.figure(figsize=(15, 9))

        pyplot.plot(step_y[-last_n:], "-", color="orange", label="Raw measurements")
        pyplot.plot(step_yhat[-last_n:], "--", color="blue", label="Predictions")
        pyplot.xlabel("Steps", fontsize=20)
        pyplot.ylabel("Efficiency", fontsize=20)
        pyplot.xticks(fontsize=24)
        pyplot.yticks(fontsize=24)
        pyplot.legend(fontsize=14, frameon=False)
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(output_path,"{jobname}-step-{step}- 700.png".format(jobname=jobname, step=i+1)))
    
    return


if __name__ == '__main__':
    config_gpu(gpu_id)
    
    try:
        df = pd.read_csv(job_path, header=[0,1], index_col=[0])
    except FileNotFoundError as e:
        logging.exception(e.errno)

    df_tmp = df[columns_to_consider[:-1]]
    df_tmp["Efficiency"] = df["Efficiency"]['mean'].values
    #df = df_tmp

    df = interpolate_df(df_tmp)

    count_nans = df.isna().sum().sum()
    if count_nans > 0:
        logging.info(f"{job_path} has {count_nans} NaNs")

    test = df.values

    test_scaled, scaler = scale_values(test)

    print(np.count_nonzero(np.isnan(test_scaled)))
    print(np.count_nonzero(np.isnan(test)))

    # reshape data
    test_x_conf_1_no_acc, test_y_conf_1_no_acc = reshape_data(
        test_scaled, n_input, n_out, input_cols_interval=(0, -1)
    )

    model = load_model(model_path)
    print(model.summary())
    print(test_x_conf_1_no_acc.shape, test_y_conf_1_no_acc.shape)

    start = time.time()
    print("Starting walk forward validation...")
    predictions, real_values = walk_forward_validation(
        model, test_x_conf_1_no_acc, test_y_conf_1_no_acc, 
        test,
        scaler,
        n_out,
        gpu_id
    )
    end = time.time()

    print("Walk forward validation done!")
    print("Elapsed time: ", end - start)
    logging.info(f"{job_path} has {end - start} test time")

    #if np.count_nonzero(np.isnan(pred_1)) > 0:
    print(f"Predictions with {np.count_nonzero(np.isnan(predictions))} NaNs")
    #if np.count_nonzero(np.isnan(real_v_1)) > 0:
    print(f"Real Values with {np.count_nonzero(np.isnan(real_values))} NaNs")


    res = []
    start = time.time()
    print("Computing metrics...")
    metrics = ['rmse', 'rmse_p', 'true_val', 'true_val_max', 'true_val_min', 'true_val_ratio_pos']
    for metric in metrics:
        score, scores = evaluate_forecast(real_values, predictions,  metric)
        res.append({'job_ID': jobname,
           'score_tot': score,
           'score_1': scores[0],
           'score_2': scores[1],
           'score_3': scores[2],
           'error_type': metric,
           'model_name': model_path,
           'experiment_name': exp_name})


    collection.insert_many(res)
    end = time.time()

    print("Metrics inserted in mongodb")
    print("Elapsed time: ", end - start)
    logging.info(f"{job_path} has {end - start} evaluation time")


    plot_results(real_values, predictions, output_path, jobname, last_n=100)
    plot_results(real_values, predictions, output_path, jobname, last_n=700)
    print("Plotted results")

    script_end = time.time()

    print("Script execution time: ", script_end - script_start)
    logging.info(f"{job_path} has {script_end - script_start} Script execution time")