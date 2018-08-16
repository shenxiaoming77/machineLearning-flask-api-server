#coding=utf-8

import redis
from  deepfm_tensorflow_server.kafkaClient import MyKafkaClient
from  deepfm_tensorflow_server.settings import *
from  deepfm_tensorflow_server.redisClient import RedisClient

# redisClient = RedisClient()
# redisClient.set('name', 'Michael')
#
# print(redisClient.get('name'))

kafkaClient = MyKafkaClient()
producer = kafkaClient.getProducer(b'deepfm_data')
for i in range(1000):
    message = 'message id :' + str(i)
    print(message)
    producer.produce(message.encode())
producer.clo
# consumer = kafkaClient.getConsumer(b'deepfm_data')
#
# for message in consumer:
#     if message is not None:
#         print (message.offset, message.value)