import json
import os
from configparser import ConfigParser

import torch

import datautils
import utils
import datetime
import time

from tasks import eval_classification
from ts2vec import TS2Vec
from ts2vec_dlinear import TS2VecDlinear

def create_model(type_of_train, dim, n_time_cols, current_device, configuration):
    if 'ts2vec-dlinear' in type_of_train.lower():
        return TS2VecDlinear(input_dims=dim, device=current_device, mode=type_of_train, n_time_cols=n_time_cols, **configuration)
    return TS2Vec(input_dims=dim, device=current_device, mode=type_of_train, **configuration)

config = ConfigParser()
config.read('config.ini')
mode = config['EXECUTION TYPE'].get('mode')
path = config['SETTINGS'].get('path')
loader = config['SETTINGS'].get('loader')
dataset = config['SETTINGS'].get('dataset')

# set GPU
device = utils.init_dl_program(0, seed=42, max_threads=8)

torch.cuda.empty_cache()

print("\n-------------- LOAD DATASET: PREPROCESSING ------------------------\n")

if loader == 'UCR':
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset)
elif loader == 'UEA':
    train_data, train_labels, test_data, test_labels = datautils.load_UEA(dataset)
else:
    raise ValueError(f"Unknown dataset")

print("Data after loading")
print("train data: " + str(train_data.shape))
print("train labels: " + str(train_labels.shape))
print("test data: " + str(test_data.shape))
print("test labels: " + str(test_labels.shape))

print("\n----------------- TRAINING ENCODER ------------------------\n")

#Creation of dirs to store results
run_dir = f'{path}/training/classification/{mode}/' + dataset + '__' + utils.name_with_datetime('classification')
os.makedirs(run_dir, exist_ok=True)

config = dict(
    batch_size=8,
    lr=0.001,
    output_dims=320,
    max_train_length=3000,
)

input_dim = train_data.shape[-1]
if mode == 'feature':
    input_dim = train_data.shape[-1] * 2

# model = TS2Vec(input_dims=train_data.shape[-1], device=device, mode=mode, **config)
model = create_model(mode, input_dim, 0, device, config)

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

out, eval_res = eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')

print("\n----------------- FINAL RESULTS --------------------\n")

utils.pkl_save(f'{run_dir}/out.pkl', out)
utils.pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
with open(f'{run_dir}/eval_res.json', 'w') as json_file:
    json.dump(eval_res, json_file, indent=4)
print('Evaluation result:', eval_res)

print("Finished.")