
# To execute the Ts2Vec model with the pipeline integration on forecasting tasks
python3 script_forecasting.py --mode ts2vec-one-loss --dataset ETTh1
python3 script_forecasting.py --mode ts2vec-one-loss --dataset ETTm1
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset WTH
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset electricity
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset traffic
pyhton3 script_forecasting.py --mode ts2vec-one-loss --dataset national_illness


# To execute the Ts2Vec original model
python3 script_forecasting.py --mode ts2vec --dataset ETTh1
python3 script_forecasting.py --mode ts2vec --dataset ETTm1
pyhton3 script_forecasting.py --mode ts2vec --dataset WTH
pyhton3 script_forecasting.py --mode ts2vec --dataset electricity
pyhton3 script_forecasting.py --mode ts2vec --dataset traffic
pyhton3 script_forecasting.py --mode ts2vec --dataset national_illness





