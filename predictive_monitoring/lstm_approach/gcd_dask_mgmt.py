import dask.dataframe as dd
from dask.delayed import delayed

import pandas as pd

import os

import statistics

def extract_delayed_dask_dataframe(data_path, df_schema, gcd_collection_name, interval_start, interval_end):
    cols = df_schema[df_schema['file pattern'] == '{gcd_coll_name}/part-?????-of-?????.csv.gz'.format(gcd_coll_name = gcd_collection_name)].content.values
    extracted_files = [os.path.join(data_path, gcd_collection_name,'part-00'+ str(v).zfill(3)+'-of-00500.csv.gz')
                        for v in range(interval_start, interval_end)]

    dfs = [delayed(pd.read_csv)(fn, header=None, index_col=False, names=cols, delimiter=',') for fn in
           extracted_files]
    dd_delayed = dd.from_delayed(dfs)
    return dd_delayed

def extract_multistepahead_dataframe(df):
    # ('mean', 'mean'), ('median', 'median'), ('min', 'min'), ('max','max'), ('q25', q25), ('q75', q75), ('q95', q95), ('std', 'std')
    aggregated_df = df.groupby("end time").agg([
        'mean',
        dd.Aggregation("median",chunk=lambda x: x.quantile(0.50), agg=lambda y: y.mean()),
        'min',
        'max',
        dd.Aggregation("q25", chunk=lambda x: x.quantile(0.25), agg=lambda y: y.mean()),
        dd.Aggregation("q75", chunk=lambda x: x.quantile(0.75), agg=lambda y: y.mean()),
        dd.Aggregation("q95", chunk=lambda x: x.quantile(0.95), agg=lambda y: y.mean()),
        #'std'
        #dd.Aggregation("std_dev", chunk=lambda x: statistics.stdev(x), agg=lambda y: statistics.stdev(y)),
    ]).compute()
    
    return aggregated_df