import numpy as np
import time

import torch

from . import _eval_protocols as eval_protocols


def generate_pred_samples_seq_len(features, data, pred_len, seq_len, drop=0):
    n = data.shape[1] - seq_len - pred_len + 1

    features = np.stack([ features[:, i:i+seq_len] for i in range(n)], axis=1)[:, 1:]
    labels = np.stack([data[:, i+seq_len:i+seq_len+pred_len] for i in range(n)], axis=1)[:, 1:]
    features = features.mean(axis=2)

    features = features[:, drop:]
    labels = labels[:, drop:]

    return features.squeeze(), labels.reshape(-1, labels.shape[2]*labels.shape[3])

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([data[:, i:1 + n + i - pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
        labels.reshape(-1, labels.shape[2] * labels.shape[3])

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }


def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, seq_len, mode, ci):
    padding = 200

    t = time.time()

    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )

    ts2vec_infer_time = time.time() - t

    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    print("Train repr shape: ", train_repr.shape)
    print("Valid repr shape: ", valid_repr.shape)
    print("Test repr shape: ", test_repr.shape)
    print("Train data shape: ", train_data.shape)
    print("Valid data shape: ", valid_data.shape)
    print("Test data shape: ", test_data.shape)
    print('----------------')

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        if seq_len is None:
            train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
            valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
            test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        else:
            train_features, train_labels = generate_pred_samples_seq_len(train_repr, train_data, pred_len, seq_len, drop=padding)
            valid_features, valid_labels = generate_pred_samples_seq_len(valid_repr, valid_data, pred_len, seq_len)
            test_features, test_labels = generate_pred_samples_seq_len(test_repr, test_data, pred_len, seq_len)

        print("train feature: ", train_features.shape)
        print("train labels: ", train_labels.shape)
        print("valid feauters: ", valid_features.shape)
        print("valid labels: ", valid_labels.shape)
        print("test features: ", test_features.shape)
        print("test labels: ", test_labels.shape)
        print("-----------------")

        t = time.time()

        if 'mlp' in mode.lower():
            lr = eval_protocols.fit_mlp(train_features, train_labels, valid_features, valid_labels)
        else:
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)

        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        # ori_shape = -1, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        print(ori_shape)
        print("Test pred: ", test_pred.shape)
        print("Test labels: ", test_labels.shape)
        print("-----------------")

        if test_data.shape[0] > 1:
            test_pred = test_pred.swapaxes(0, 3)
            test_pred = test_pred.squeeze(0)
            test_pred = test_pred.reshape(test_pred.shape[0] * test_pred.shape[1], test_pred.shape[2])

            test_labels = test_labels.swapaxes(0, 3)
            test_labels = test_labels.squeeze(0)
            test_labels = test_labels.reshape(test_labels.shape[0]*test_labels.shape[1], test_labels.shape[2])

            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)

            # test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
            # test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            test_pred = test_pred.squeeze(0)
            test_pred = test_pred.reshape(test_pred.shape[0] * test_pred.shape[1], test_pred.shape[2])

            test_labels = test_labels.squeeze(0)
            test_labels = test_labels.reshape(test_labels.shape[0] * test_labels.shape[1], test_labels.shape[2])

            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)

        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
