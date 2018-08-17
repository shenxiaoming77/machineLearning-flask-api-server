#coding=utf-8

import  ast
import redis
from  deepfm_tensorflow_server.kafkaClient import Kafka_Producer
from  deepfm_tensorflow_server.settings import *
from  deepfm_tensorflow_server.redisClient import RedisClient

# redisClient = RedisClient()
# redisClient.set('name', 'Michael')
#
# print(redisClient.get('name'))


# producer = Kafka_Producer('deepfm_data')
#
# for i in range(1000000):
#     message = 'message: ' + str(i)
#     print(message)
#     producer.send(message)

s = '{"host":"192.168.11.22", "data" :{"port":3306, "user":"abc", "passwd":"123", "db":"mydb", "connect_timeout":10}}'

d = ast.literal_eval(s)

print(d['data'])