import json
import sys

import requests
from sklearn.preprocessing import MinMaxScaler

from gcd_data_manipulation import data_aggregation
from gcd_data_manipulation import load_data
from gcd_data_manipulation import series_to_supervised


job_id = int(sys.argv[1])
name = "LSTM_on_test_workload_%i" % job_id



columns_to_consider = ['end time',
                       'CPU rate',
                       'canonical memory usage',
                       'assigned memory usage',
                       'unmapped page cache',
                       'total page cache',
                       'maximum memory usage',
                       'disk I/O time',
                       'local disk space usage',
                       'maximum CPU rate',
                       'maximum disk IO time',
                       'cycles per instruction',
                       'memory accesses per instruction',
                       'CPU ratio usage',
                       'memory ratio usage',
                       'disk ratio usage',
                       'Efficiency'  # target metric
                       ]

readings_df = load_data('task-usage_job-ID-%i_total.csv' % job_id, columns_to_consider)
readings_df = data_aggregation(readings_df, aggr_type='mean')

values = readings_df.values

values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[i for i in range(int(reframed.shape[1] / 2), (2 * int(reframed.shape[1] / 2)) - 1)]],
              axis=1,
              inplace=True)

# split into train and test sets
values = reframed.values

test_X, test_y = values[:1, :-1], values[:, -1]

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

SERVER_URL = 'http://cost-efficiency-serving-service:8500/v1/models/cost-efficiency:predict'
SERVER_URL = 'http://localhost:8501/v1/models/cost:predict'

predict_request = json.dumps({test_X.tolist})
response = requests.post(SERVER_URL, data=predict_request)
print(response.json())
