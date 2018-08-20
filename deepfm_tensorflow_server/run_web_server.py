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
from deepfm_tensorflow_server.myencoder import MyEncoder

from deepfm_tensorflow_server.settings import *


#initialize the flask application and redis server
app = flask.Flask(__name__)
app.json_encoder = MyEncoder

redis_db = redis.StrictRedis(host = REDIS_HOST,
                             port = REDIS_PORT,
                             db = REDIS_DB)




@app.route("/api/predict", methods = ["POST"])
def predict():
    result = {}

    if flask.request.method == "POST":
        param = request.get_data()
        print(param)
        param_json = json.loads(param)
        id = param_json['id']
        redis_db.rpush(MESSAGE_QUEUE, param)

        while True:
            output = redis_db.get(id)
            if output is not None:
                result[id] = output
                redis_db.delete(id)
                break
            time.sleep(CLIENT_SLEEP)

    return  flask.jsonify(result)

if __name__ == '__main__':

    app.run(host="127.0.0.1", port=5000, debug=True)