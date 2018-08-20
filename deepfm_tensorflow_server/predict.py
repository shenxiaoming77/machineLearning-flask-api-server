#coding=utf-8



import  sys
import  os
import tensorflow as tf
import  numpy as np
import  pandas as pd
import  logging
import  pickle


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from deepfm_tensorflow_server.settings import *
from  deepfm_tensorflow_server.DeepFM import DeepFM
from  deepfm_tensorflow_server.utilities import *

'''
tf.app.flags.DEFINE_xxx()就是添加命令行的optional argument（可选参数）
tf.app.flags.FLAGS可以从对应的命令行参数取出参数
'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('batch_size', 512, 'number of batch size')
tf.app.flags.DEFINE_float('l1_reg', 2e-3, 'L1 regularization')
tf.app.flags.DEFINE_float('l2_reg', 0.0001, 'L2 regularization')
tf.app.flags.DEFINE_float("embedding_size", 40, "Embedding size")

class DeepFMEstimator:

    def check_restore_parameters(self, sess, saver):
        """ Restore the previously trained parameters if there are any. """
        ckpt = tf.train.get_checkpoint_state("checkpoints")
        if ckpt and ckpt.model_checkpoint_path:
            logging.info("Loading parameters for the my CNN architectures...")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.error("there is no model file in model checkpoint paht.....")


    def batch_predict(self, sess, model, print_every = 50):
        """training model"""
        # get testing data, iterable
        test_data = pd.read_csv(Root_Dir + 'test.csv',
                                chunksize=model.batch_size)
        test_step = 1
        # batch_size data
        for data in test_data:
            actual_batch_size = len(data)
            batch_X = []
            batch_idx = []
            for i in range(actual_batch_size):
                sample = data.iloc[i,:]
                array,idx = one_hot_representation(sample, fields_test_dict, test_array_length)
                batch_X.append(array)
                batch_idx.append(idx)

            batch_X = np.array(batch_X)
            batch_idx = np.array(batch_idx)

            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_X,
                        model.keep_prob:1,
                        model.feature_index:batch_idx}

            # shape of [None,2]
            y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
            # write to csv files

            data['click'] = y_out_prob[0][:, -1]  #is a tensor, tf.shape(y_out_prob): [1, 512, 2]
            print(y_out_prob[0][:, 0])
            if test_step == 1:
                data[['id','click']].to_csv('Deep_FM_FTRL_v1.csv', mode='a', index=False, header=True)
            else:
                data[['id','click']].to_csv('Deep_FM_FTRL_v1.csv', mode='a', index=False, header=False)

            test_step += 1
            if test_step % 50 == 0:
                logging.info("Iteration {0} has finished".format(test_step))



# seting fields
fields_test = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                   'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                   'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
                   'device_conn_type']
# loading dicts
fields_test_dict = {}
for field in fields_test:
    with open('dicts/'+field+'.pkl','rb') as f:
        fields_test_dict[field] = pickle.load(f)

with open('dicts/' + 'click.pkl', 'rb') as f:
    fields_test_dict['click'] = pickle.load(f)

# length of representation
feature_length = max(fields_test_dict['click'].values()) - 1

fields_test_dict.pop('click')

# initialize the model
config = {}
config['lr'] = 0.01
config['batch_size'] = 512
config['reg_l1'] = 2e-3
config['reg_l2'] = 0
config['k'] = 40

# get feature length
config['feature_length'] = feature_length
test_array_length = feature_length
print('feature length:' , config['feature_length'])

# num of fields
field_cnt = 21
config['field_cnt'] = field_cnt

model = DeepFM(config)

# build graph for model
model.build_graph()

saver = tf.train.Saver(max_to_keep=5)

predictor = DeepFMEstimator()

with tf.Session() as sess:
    # TODO: with every epoches, print training accuracy and validation accuracy
    #sess.run(tf.global_variables_initializer())

    # restore trained parameters
    print('restore trained model parameters.....')
    predictor.check_restore_parameters(sess, saver)

    print('start perdict...')
    predictor.batch_predict(sess, model)




