#coding=utf-8

from kafka import  KafkaConsumer
from  deepfm_tensorflow_server.settings import *

topic = 'deepfm_data'
consumer = KafkaConsumer(topic,

                         bootstrap_servers = ['10.8.26.23:9092']
                        )
print('start to consume message')
i = 0
try:
    for message in consumer:
        print(i)
        print(message.value)
        i = i + 1
except Exception as e:
    print(e)