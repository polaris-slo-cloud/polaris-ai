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

job_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]
mongodb_name = sys.argv[4] #default predictiveDB
mongodb_collection = sys.argv[5] #default multistepahead
n_input = int(sys.argv[6])
n_out = int(sys.argv[7])

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

df = pd.read_csv(job_path, header=[0,1], index_col=[0])
df_tmp = df[columns_to_consider[:-1]]
df_tmp["Efficiency"] = df["Efficiency"]['mean']
df = df_tmp
test = df.values


test_scaled, scaler = scale_values(test)
# reshape data
test_x_conf_1_no_acc, test_y_conf_1_no_acc = reshape_data(
    test_scaled, n_input, n_out, input_cols_interval=(0, -1)
)

model = load_model(model_path)
print(model.summary())
print(test_x_conf_1_no_acc.shape, test_y_conf_1_no_acc.shape)

print("Starting walk forward validation...")
score_rmse, scores_rmse, pred_1, real_v_1 = walk_forward_validation(
    model, test_x_conf_1_no_acc, test_y_conf_1_no_acc, 
    test,
    scaler,
    n_out
)
print("Walk forward validation done!")
      
res = [{'job_ID': jobname,
       'score_tot': score_rmse,
       'score_1': scores_rmse[0],
       'score_2': scores_rmse[1],
       'score_3': scores_rmse[2],
       'error_type': 'rmse',
      'model_name': model_path,
      'experiment_name': exp_name}]


print("Computing metrics...")
metrics = ['rmse_p', 'true_val', 'true_val_max', 'true_val_min', 'true_val_ratio_pos']
for metric in metrics:
    score, scores = evaluate_forecast(real_v_1, pred_1,  metric)
    res.append({'job_ID': jobname,
       'score_tot': score,
       'score_1': scores[0],
       'score_2': scores[1],
       'score_3': scores[2],
       'error_type': metric,
       'model_name': model_path,
       'experiment_name': exp_name})

    
collection.insert_many(res)
print("Metrics inserted in mongodb")

for i in range(3):
    step_y = [x[0] for x in real_v_1]
    step_yhat = [x[0] for x in pred_1]

    pyplot.clf()
    pyplot.figure(figsize=(15, 9))

    pyplot.plot(step_y[-700:], "-", color="orange", label="Raw measurements")
    pyplot.plot(step_yhat[-700:], "--", color="blue", label="Predictions")
    pyplot.xlabel("Steps", fontsize=20)
    pyplot.ylabel("Efficiency", fontsize=20)
    pyplot.xticks(fontsize=24)
    pyplot.yticks(fontsize=24)
    pyplot.legend(fontsize=14, frameon=False)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(output_path,"{jobname}-step-{step}-700.png".format(jobname=jobname, step=i+1)))

print("Plotted results")
