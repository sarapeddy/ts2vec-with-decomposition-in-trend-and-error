# TS2Vec with decomposition in trend and error

This repository contains the implementation of our proposed pipeline with Ts2Vec [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) (AAAI-22). 

## Requirements

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate TS2Vec on a dataset for forecasting task, run the following command:

```train & evaluate
python3 script_forecasting.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/forecasting/<mode>/DatasetName__RunName_Date_Time/`. 

