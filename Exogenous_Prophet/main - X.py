import dataBringer
import dataPrepareator
import os
import pandas as pd
import data
import json
from prophet import Prophet
from dbConn import DbConnection
import warnings
warnings.filterwarnings('ignore')

def getCSVfileList():
    files = os.listdir()
    files = [x for x in files if x.endswith('csv')]
    return files


files = getCSVfileList()
if len(files) > 0:
    print('Found csv in folder. Deleting.')
    for x in files:
        os.remove(x) 

conn = DbConnection(username= data.db_username, password= data.db_password, host = data.db_host, port= data.db_port, name = data.db_name)

dataBringer.downloadDataForecast()
files = getCSVfileList()

data_epias = dataBringer.getDataTrainFromEpias()

data_metalogica = pd.DataFrame()
for file in files:

    df_sub = pd.read_csv(file)
    data_metalogica = pd.concat([data_metalogica, df_sub], axis = 1)

    
df_train, df_test = dataPrepareator.train_test_split(data_epias=data_epias, data_metalogica = data_metalogica)

print('Train & pred ready.')

with open('paramsMay.txt') as f:
    params = f.read()
    params = json.loads(params)
params = params['params']

print('Read hyperparameters.')

print('Training started.')
final_model = Prophet(changepoint_prior_scale= params['changepoint_prior_scale'],
                      holidays_prior_scale = params['holidays_prior_scale'],
                      n_changepoints = 350,
                      seasonality_mode = 'multiplicative',
                      weekly_seasonality=params['weekly_seasonality'],
                      daily_seasonality = params['daily_seasonality'],
                      yearly_seasonality = False,
                      growth = 'linear',
                      interval_width=0.95)
for i in df_train.columns[2:].tolist():
    final_model.add_regressor(i)
final_model.fit(df_train)
print('Predicting future.')
pred = final_model.predict(df_test)
pred = pred[['ds', 'yhat']]
print('Cleanup started.')
for x in files:
        os.remove(x) 
print('Predictions inserting into database.')
conn.dataToDB(df = pred,  schema ='Example',table = 'Exp_Frcst', if_there = 'append', index = False)


print('Finish.')
