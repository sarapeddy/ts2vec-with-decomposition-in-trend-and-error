import json

import torch.cuda

from tasks import eval_anomaly_detection, eval_anomaly_detection_coldstart
from ts2vec import TS2Vec
import datautils
import utils
import os
import time
import datetime
from configparser import ConfigParser
from ts2vec_dlinear import TS2VecDlinear


def create_model(type_of_train, dim, n_time_cols, current_device, configuration):
    if 'ts2vec-dlinear' in type_of_train.lower():
        return TS2VecDlinear(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)
    return TS2Vec(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)


# To configure the path to store the files and the dataset
config = ConfigParser()
config.read('config_anomaly_detection.ini')
mode = config['EXECUTION TYPE'].get('mode')
path = config['SETTINGS'].get('path')
loader = config['SETTINGS'].get('loader')
dataset = config['SETTINGS'].get('dataset')
ci = config['PARAMETERS'].getboolean('ci')

# set GPU
device = utils.init_dl_program(0, seed=42, max_threads=8)

torch.cuda.empty_cache()

print("\n-------------- LOAD DATASET: PREPROCESSING ------------------------\n")

if loader == 'anomaly':
    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(dataset)
    train_data = datautils.gen_ano_train_data(all_train_data)
elif loader == 'anomaly_coldstart':
    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(dataset)
    train_data, _, _, _ = datautils.load_UCR('FordA')
else:
    raise ValueError(f"Unknown loader")

print("Data after loading")
print("train data: " + str(train_data.shape))

print("\n----------------- TRAINING ENCODER ------------------------\n")

#Creation of dirs to store results
run_dir = f'{path}/training/anomaly_detection/{loader}/{mode}/' + dataset + '__' + utils.name_with_datetime('anomaly_detection')
os.makedirs(run_dir, exist_ok=True)

if ci:
    config = dict(
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000
    )

    input_dim = train_data.shape[-1]
    if mode == 'feature':
        input_dim = train_data.shape[-1] * 2

    # model = TS2Vec(input_dims=train_data.shape[-1], device=device, mode=mode, **config)
    model = create_model(mode, input_dim, 0, device, config)

else:
    config = dict(
        batch_size=1,
        lr=0.001,
        output_dims=40,
        max_train_length=3000
    )

    # model = TS2Vec(input_dims=train_data.shape[-1], device=device, mode=mode, **config)
    model = create_model(mode, 1, 0, device, config)

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

print("\n----------------- EVAL CLASSIFICATION -------------------\n")

if loader == 'anomaly':
    out, eval_res = eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay, ci)
elif loader == 'anomaly_coldstart':
    out, eval_res = eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay, ci)
else:
    raise ValueError(f"Unknown loader")

print("\n----------------- FINAL RESULTS --------------------\n")

utils.pkl_save(f'{run_dir}/out.pkl', out)
utils.pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
with open(f'{run_dir}/eval_res.json', 'w') as json_file:
    json.dump(eval_res, json_file, indent=4)
print('Evaluation result:', eval_res)

print("Finished.")