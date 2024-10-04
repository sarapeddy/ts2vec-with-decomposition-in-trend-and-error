# TS2Vec: implementation with 1 - step pipeline

<p align="center">
  <img src="/images/ts2vec.png" alt="ts2vec" width="600" />
</p> 

## Requirements

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate TS2Vec on a dataset for forecasting task, run the following command:

```sh
sh forecasting.sh
```

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/forecasting/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`.

To train and evaluate TS2Vec on a dataset for classfication task, run the following command:

```sh
sh classication.sh
```

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/classication/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`. 

