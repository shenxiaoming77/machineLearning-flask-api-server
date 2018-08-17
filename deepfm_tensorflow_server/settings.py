Root_Dir = 'D:/conf_test/ctr_prediction/avazu_ctr/'

# seting fields
fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
          'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
          'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
          'device_conn_type']

#redis config
REDIS_HOST = "10.8.26.26"
REDIS_PORT = 6379
REDIS_DB = 0

#kafka config
KAFKA_HOSTS = "10.8.26.23:9092,10.8.26.24:9092,10.8.26.25:9092"


#initialize the constants
MESSAGE_QUEUE = 'deepfm_data'
BATCH_SIZE  = 32
SERVER_SLEEP = 0.25