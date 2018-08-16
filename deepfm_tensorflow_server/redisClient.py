import  redis

from  .settings import *

class RedisClient:

    def __init__(self):
        db = redis.StrictRedis(host = REDIS_HOST,
                               port = REDIS_PORT,
                               db = REDIS_DB)