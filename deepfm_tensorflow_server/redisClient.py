import  redis

from  .settings import *

class RedisClient:

    def __init__(self):
        self.redis = redis.StrictRedis(host = REDIS_HOST,
                                       port = REDIS_PORT,
                                       db = REDIS_DB)

    def get(self, key):
        return self.redis.get(key)

    def set(self, key, value):
        return  self.redis.set(key, value)