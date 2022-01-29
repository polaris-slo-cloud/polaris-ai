import pickle as pkl

import dask.dataframe as dd
import pandas as pd
import os
import glob

from gcd_dask_mgmt import extract_multistepahead_dataframe

SOURCE_DIR = "/home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/scheduling_class_3_29days/"
TARGET_DIR = "/home/tuwien/data/cloud_data/Google-clusterdata-2011-2/processed_data/high-level_monitoring/aggregated_multistep_tests/"
COLUMNS_TO_CONSIDER = [
    "end time",
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
METRICS = ['mean', 'median', 'min', 'max', 'q25', 'q75', 'q95','std']


all_files = os.listdir(SOURCE_DIR)

transformer_files = os.listdir('/home/tuwien/vcasamayor/results/all_multistep/')
computed_files = os.listdir(TARGET_DIR)

final_set = [f for f in transformer_files if f not in computed_files]
print(len(final_set))
for i, f in enumerate(final_set):
    perc = round((i / len(final_set) * 100))
    if perc % 5 == 0:
        print(f"{perc}% processed")
    df = pd.read_csv(os.path.join(SOURCE_DIR, f), index_col=0)
    df = df[COLUMNS_TO_CONSIDER]
    df["datetime"] = timestamps_readings = [
        pd.Timestamp(int(t / 1000000) + 1304233200, tz="US/Pacific", unit="s")
        for t in df["end time"].values
    ]
    df.set_index("datetime", inplace=True)
    df = df.interpolate(method="time").ffill().bfill()
    print(df.isnull().sum().sum())
    ddf = dd.from_pandas(df, npartitions = 4)
    std_df = ddf.groupby('end time').std().compute()
    agg_df = extract_multistepahead_dataframe(ddf)
    for col in COLUMNS_TO_CONSIDER[1:]:
        agg_df[(col, 'std')] = std_df[col].values

    final_df = agg_df[COLUMNS_TO_CONSIDER[1]]
    cols = list(zip([COLUMNS_TO_CONSIDER[1]] * len(METRICS), METRICS))
    final_df.columns = pd.MultiIndex.from_tuples(cols)

    for col in COLUMNS_TO_CONSIDER[2:]:
        tmp_df = agg_df[col]
        cols = list(zip([col]*len(METRICS), METRICS))
        tmp_df.columns = pd.MultiIndex.from_tuples(cols)
        final_df = final_df.join(tmp_df, on = "end time")
        
    final_df.to_csv(os.path.join(TARGET_DIR, f))

    
computed_filepaths = glob.glob(os.path.join(TARGET_DIR, 'task-usage_job-ID-*'))
    
chunks = [computed_filepaths[i:i + 100] for i in range(0, len(computed_filepaths), 100)]

for i, c in enumerate(chunks):
    with open(os.path.join(TARGET_DIR, f'aggregated_list_0{i}_of_05.txt'), 'w') as f:
        f.writelines('\n'.join(c))
