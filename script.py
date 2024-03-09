import json

import torch.cuda
from ts2vec import TS2Vec
import datautils
import utils
import os
import time
import datetime
import pandas as pd
from configparser import ConfigParser
from tasks.forecasting import eval_forecasting

# RICORDATI DI MODIFICARE QUANDO HAI CREATO LA NUOVA CLASSE
def create_model(type_of_train, dim, current_device, configuration):
    if type_of_train == 'ts2vec-Dlinear':
        return TS2Vec(input_dims=dim, device=current_device, **configuration)
    else:
        return TS2Vec(input_dims=dim, device=current_device, **configuration)


# To configure the path to store the files and the dataset
config = ConfigParser()
config.read('config.ini')
mode = config['EXECUTION TYPE'].get('mode')
path = config['SETTINGS'].get('path')
dataset = config['SETTINGS'].get('dataset')

# To extract the csv from the electricity dataset: it is downloaded as a txt file named as LD2011_2014
if dataset == 'LD2011_2014':
    data_ecl = pd.read_csv(f'datasets/{dataset}.txt', parse_dates=True, sep=';', decimal=',', index_col=0)
    data_ecl = data_ecl.resample('1h', closed='right').sum()
    data_ecl = data_ecl.loc[:, data_ecl.cumsum(axis=0).iloc[8920] != 0]  # filter out instances with missing values
    data_ecl.index = data_ecl.index.rename('date')
    data_ecl = data_ecl['2012':]
    data_ecl.to_csv(f'datasets/{dataset}.csv')
    dataset = 'electricity'

# set GPU
device = utils.init_dl_program(0, seed=42, max_threads=8)

torch.cuda.empty_cache()

print("-------------- LOAD DATASET: PREPROCESSING ------------------------")

data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols = datautils.load_forecast_csv(dataset)
# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

print("Data after StandarScaler on n_coviariate_cols and original features")
print(data)
print(data.shape)
print("train:", train_slice)
print("valid: ", valid_slice)
print("test:", test_slice)
print("pred_lens: ", pred_lens)
print("n_time_cols:", n_time_cols)
print('------------')
print(scaler)

train_data = data[:, train_slice]
print(train_data)
print(train_data.shape)

config = dict(
    batch_size=8,
    lr=0.001,
    output_dims=320,
    max_train_length=3000
)

#Creation of dirs to store results
run_dir = f'{path}/training/' + dataset + '__' + utils.name_with_datetime('forecast_multivar')
os.makedirs(run_dir, exist_ok=True)

print("\n------------------- TRAINING ENCODER -------------------\n")

input_dim = train_data.shape[-1]
if mode == 'ts2vec-feature':
    input_dim = train_data.shape[1] + train_data.shape[-1] - n_time_cols

t = time.time()

# Train a TS2Vec model
model = create_model(mode, input_dim, device, config)

loss_log = model.fit(
    train_data,
    n_epochs=None,
    n_iters=None,
    verbose=True
)

model.save(f'{run_dir}/model.pkl')

t = time.time() - t
print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

print("\n----------------- EVAL FORECASTING -------------------\n")

out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols)

print("\n----------------- FINAL RESULTS --------------------\n")

utils.pkl_save(f'{run_dir}/out.pkl', out)
utils.pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
with open(f'{run_dir}/eval_res.json', 'w') as json_file:
    json.dump(eval_res, json_file, indent=4)

print('Evaluation result:', eval_res)

torch.cuda.empty_cache()

print("Finished")
