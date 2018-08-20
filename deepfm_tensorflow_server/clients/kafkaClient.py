#coding=utf-8
# -*- coding: utf-8 -*-

'''
    使用kafka-Python 1.3.3模块
'''

import sys
import time
import json

from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from  .settings import *

class Kafka_Producer():
    '''
    生产模块：根据不同的key，区分消息
    '''

    def __init__(self, topic):
        self.producer = KafkaProducer(bootstrap_servers = KAFKA_HOSTS)
        self.kafkatopic = topic

    def send(self,key, message):
        try:
            producer = self.producer
            producer.send(self.kafkatopic, key = key, value = message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print (e)

    def send(self, message):
        try:
            producer = self.producer
            producer.send(self.kafkatopic, value = message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print (e)






# def main(xtype, group, key):
#     '''
#     测试consumer和producer
#     '''
#     if xtype == "p":
#         # 生产模块
#         producer = Kafka_producer(KAFAKA_HOST)
#         print ("===========> producer:", producer)
#         for _id in range(100):
#            params = '{"msg" : "%s"}' % str(_id)
#            producer.sendjsondata(params)
#            time.sleep(1)
#
#     if xtype == 'c':
#         # 消费模块
#         consumer = Kafka_consumer(KAFAKA_HOST, KAFAKA_PORT, KAFAKA_TOPIC, group)
#         print "===========> consumer:", consumer
#         message = consumer.consume_data()
#         for msg in message:
#             print 'msg---------------->', msg
#             print 'key---------------->', msg.key
#             print 'offset---------------->', msg.offset
#
#
#
# if __name__ == '__main__':
#     xtype = sys.argv[1]
#     group = sys.argv[2]
#     key = sys.argv[3]
#     main(xtype, group, key)
