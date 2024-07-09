# TS2Vec with decomposition in trend and error

This repository contains the implementation of a revisitation of the paper [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466) (AAAI-22). The time series, indeed, are decomposed in two components: trend and error. Each of them is then cropped and passed into an encoder (the same of TS2vec). The architecture can be seen in the figure. 

<div style="display: flex; justify-content: center;">
  <img src="/images/ts2vec_architecture.png" alt="Logo" width="500"/>
</div>



## Requirements

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/ETTh1.csv`, `datasets/ETTh2.csv` and `datasets/ETTm1.csv`.
* [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) should be preprocessed using `datasets/preprocess_electricity.py` and placed at `datasets/electricity.csv`.

## Usage

To train and evaluate TS2Vec on a dataset, run the following command:

```train & evaluate
python script_forecasting.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/forecasting/<mode>/DatasetName__RunName_Date_Time/`. 

