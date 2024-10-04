import argparse
import json

import torch.cuda

from tasks import eval_classification
from ts2vec import TS2Vec
import datautils
import utils
import os
import time
import datetime
import pandas as pd
from configparser import ConfigParser
from tasks.forecasting import eval_forecasting
from ts2vec_ablation import TS2VecAblation
from ts2vec_dlinear import TS2VecDlinear
from dlinear_exp import DLinear


def create_model(type_of_train, dim, n_time_cols, current_device, configuration):
    if 'ts2vec-dlinear' in type_of_train.lower():
        return TS2VecDlinear(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)
    return TS2Vec(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)


# To configure the path to store the files and the dataset
# config = ConfigParser()
# config.read('config_forecasting.ini')
# mode = config['EXECUTION TYPE'].get('mode')
# path = config['SETTINGS'].get('path')
# dataset = config['SETTINGS'].get('dataset')
# ci = config['PARAMETERS'].getboolean('ci')
# seq_len = config['PARAMETERS'].get('seq_len')

config = argparse.ArgumentParser()
config.add_argument('--mode', type=str, default='ts2vec-Dlinear-one-loss')
config.add_argument('--path', type=str, default='/home/n269362/projects/ts2vec')
config.add_argument('--dataset', type=str, default='WTH')
config.add_argument('--ci', action='store_true', default=False)
config.add_argument('--seq_len', type=int, default=None)

args = config.parse_args()
mode = args.mode
path = args.path
dataset = args.dataset
ci = args.ci
seq_len = args.seq_len


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
valid_data = data[:, valid_slice]
test_data = data[:, test_slice]
print(train_data)
print(train_data.shape)

#Creation of dirs to store results
string_seq_len = f'seq_len_{seq_len}' if seq_len else 'normal'
run_dir = f'{path}/training/forecasting/{string_seq_len}/{mode}/' + dataset + '__' + utils.name_with_datetime('forecast_multivar')
os.makedirs(run_dir, exist_ok=True)

if mode.lower() != 'DLinear'.lower():

    print("\n------------------- TRAINING ENCODER -------------------\n")

    if not ci:
        input_dim = train_data.shape[-1]
        # input_dim = train_data.shape[-1] - n_time_cols
        if mode == 'feature':
            input_dim = train_data.shape[-1] + train_data.shape[-1] - n_time_cols

        config = dict(
            batch_size=8,
            lr=0.001,
            output_dims=320,
            max_train_length=3000,
            ci=ci
        )

        # Train a TS2Vec model
        model = create_model(mode, input_dim, n_time_cols, device, config)
        # model = TS2VecAblation(input_dims=input_dim, device=device, mode='ts2vec-ablation-err', n_time_cols=n_time_cols, **config)
    else:
        config = dict(
            batch_size=4,
            lr=0.001,
            output_dims=32,
            max_train_length=3000,
            ci=ci
        )

        # Train a TS2Vec model
        model = create_model(mode, 1, n_time_cols, device, config)

    t = time.time()

    loss_log = model.fit(
        train_data,
        n_epochs=None,
        n_iters=None,
        verbose=True
    )

    if 'ts2vec-dlinear' in mode.lower():
        model.save(f'{run_dir}/model_avg.pkl', f'{run_dir}/model_err.pkl')
    else:
        model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("\n----------------- EVAL FORECASTING -------------------\n")

    out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols, seq_len, mode, ci)

    print("\n----------------- FINAL RESULTS --------------------\n")

    utils.pkl_save(f'{run_dir}/out.pkl', out)
    utils.pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
    with open(f'{run_dir}/eval_res.json', 'w') as json_file:
        json.dump(eval_res, json_file, indent=4)

    print('Evaluation result:', eval_res)

else:
    print("\n----------------- EVAL FORECASTING -------------------\n")

    results = {'ours': {}}
    for pred_len in pred_lens:
        config = dict(
            name_dataset = dataset,
            seq_len=seq_len if seq_len else 96,
            pred_len=pred_len,
            enc_in=data.shape[-1]-n_time_cols,
            individual=False,
            lr = 0.001,
            batch_size = 16 if dataset == 'electricity' else 8,
        )

        model = DLinear(device, n_time_cols, run_dir, **config)

        print(f"Training DLinear for pred_len = {pred_len}")

        model.train(train_data, valid_data, test_data, scaler)

        print(f"Testing DLinear for pred_len = {pred_len}")

        results = model.test(test_data, scaler, results=results)

    print("\n----------------- FINAL RESULTS --------------------\n")

    print(results)
    with open(f'{run_dir}/eval_res.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

torch.cuda.empty_cache()

print("Finished")
