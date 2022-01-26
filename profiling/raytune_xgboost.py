import pandas as pd

import dask.dataframe as dd
from dask.delayed import delayed

import numpy as np
import random

import os
from collections import Counter

from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


from sklearn.model_selection import train_test_split
import sklearn.metrics
import os

from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

import seaborn as sns
import xgboost as xgb


static_metrics = pd.read_csv('data/static_metrics_and_kmeans_v1.csv')

print("Input-Dataframe")
print(static_metrics.head(10))

oneHotDf = static_metrics[['different machines restriction', 'disk space request - Quartiles', 'memory request - Quartiles', 'CPU request - Quartiles', 'priority labels', 'user', 'logical job name', 'scheduling class']]

oneHotDf = pd.get_dummies(oneHotDf)
oneHotDf.columns = [col.replace('[', '').replace(']','').replace(',',' ').replace(' ', '_') for col in oneHotDf.columns]

print("One-Hot-Dataframe")
print(oneHotDf)

def train_xgboost_cluster(config: dict, data=None):
    # This is a simple training function to be passed into Tune
    # Load dataset
    data, labels = data
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier, using the Tune callback
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])

    
def get_best_model_checkpoint(analysis):
    best_bst = xgb.Booster()
    best_bst.load_model(os.path.join(analysis.best_checkpoint, "model.xgb"))
    accuracy = 1. - analysis.best_result["eval-error"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst

def tune_xgboost(data):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        #"tree_method": "gpu_hist",
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1)
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=10,  # 10 training iterations
        grace_period=1,
        reduction_factor=2)

    analysis = tune.run(
        tune.with_parameters(train_xgboost_cluster, data=data),
        #fail_fast="raise", # to debug
        metric="eval-logloss",
        mode="min",
        resources_per_trial={"cpu": 8}, # "gpu":0.1},
        config=search_space,
        num_samples=800,
        scheduler=scheduler,
        raise_on_failed_trial = False)

    return analysis



analysis = tune_xgboost([oneHotDf, (static_metrics['K-Means = 4'] == 0).astype(int).values])

best_bst = get_best_model_checkpoint(analysis)