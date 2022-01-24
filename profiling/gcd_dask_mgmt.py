import dask.dataframe as dd
from dask.delayed import delayed

import pandas as pd

import os


def extract_delayed_dask_dataframe(data_path, df_schema, gcd_collection_name, interval_start, interval_end):
    cols = df_schema[df_schema['file pattern'] == '{gcd_coll_name}/part-?????-of-?????.csv.gz'.format(gcd_coll_name = gcd_collection_name)].content.values
    extracted_files = [os.path.join(data_path, gcd_collection_name,'part-00'+ str(v).zfill(3)+'-of-00500.csv.gz')
                        for v in range(interval_start, interval_end)]

    dfs = [delayed(pd.read_csv)(fn, header=None, index_col=False, names=cols, delimiter=',') for fn in
           extracted_files]
    dd_delayed = dd.from_delayed(dfs)
    return dd_delayed