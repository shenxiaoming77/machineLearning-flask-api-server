import  pandas as pd
import  flask
from flask import Flask, jsonify
from  flask import  abort
from  flask import  make_response
from  flask import  request
import  numpy as np
import  random

import  redis
import  json
import  time

data_df = pd.read_csv('D:/conf_test/ctr_prediction/avazu_ctr/test.csv')
columns_list = data_df.columns

def get_data(index):
    sample = data_df.iloc[index,:]
    return  sample

def prepare_data():
    index = random.randint(1, 100000)
    print(index)
    sample_df = get_data(index)
    data_dict = {}
    for key in [key for key in columns_list if key  not in ['id']]:
        print(key, '  ', sample_df[key])
        data_dict[key] = sample_df[key]

    return  {'id':index, 'data':data_dict}

def predict():
    sample = prepare_data()
    data_json_str = json.dumps(sample)
    print(data_json_str)
    r = request.post('http://127.0.0.1:5000/api/predict', data = data_json_str)
    result = json.loads(r.text)
    print(result)

if __name__ == '__main__':
    predict()