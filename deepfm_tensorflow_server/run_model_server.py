#coding=utf-8
import  redis
from deepfm_tensorflow_server.estimator import DeepFMEstimator
import  numpy as np
from  deepfm_tensorflow_server.settings import  *
import  ast
import  pandas as pd
import  time
import  json

redis_db = redis.StrictRedis(host = REDIS_HOST,
                             port = REDIS_PORT,
                             db = REDIS_DB)



def _init_config():
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 512
    config['reg_l1'] = 2e-3
    config['reg_l2'] = 0
    config['k'] = 40

    # get feature length
    config['feature_length'] = 9041

    # num of fields
    field_cnt = 21
    config['field_cnt'] = field_cnt

    return  config


def start_run_server(config):
    print('loading deepfm model')
    estimator = DeepFMEstimator(config)
    print('deepfm model loaded')

    while True:
        queue = redis_db.lrange(MESSAGE_QUEUE, 0, BATCH_SIZE - 1)
        ids = []
        message_list = []
        for message in queue:

            if message is not  None:
                #message_dict = ast.literal_eval(message)
                message_dict = json.loads(message)
                message_list.append(message_dict)

        if len(message_list) > 0:
            for message_dict in message_list:
                id = message_dict['id']
                data_dict = message_dict['data']
                prob = estimator.predict(data_dict)[0]
                print('id:', id, '  prob:', prob)
                redis_db.set(id, prob)
                ids.append(id)

            # remove the set of data from our queue
            redis_db.ltrim(MESSAGE_QUEUE, len(ids), -1)

        # sleep for a small amount
        time.sleep(SERVER_SLEEP)


if __name__ == '__main__':
    print('initial config for deepfm model')
    config = _init_config()

    print('start to run web server')
    start_run_server(config)