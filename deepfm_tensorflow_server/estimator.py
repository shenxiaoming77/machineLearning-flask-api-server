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

from .settings import *
from  .DeepFM import DeepFM
from  .utilities import one_hot_representation



class DeepFMEstimator:

    def __init__(self, config):

        self.config = config

        self.model = DeepFM(config)
        # build graph for model
        self.model.build_graph()

        self.saver = tf.train.Saver(max_to_keep=5)

        self.session = tf.Session()

        # restore trained parameters
        print('restore trained model parameters.....')
        self.check_restore_parameters(self.session, self.saver)

        #load field dicts
        self.fields_predict_dict = {}
        for field in fields:
            with open('dicts/'+field+'.pkl','rb') as f:
                self.fields_predict_dict[field] = pickle.load(f)

    def update_session(self, session):
        #self.session = session
        self.session = tf.Session()

    def check_restore_parameters(self, sess, saver):
        """ Restore the previously trained parameters if there are any. """
        ckpt = tf.train.get_checkpoint_state("checkpoints")
        if ckpt and ckpt.model_checkpoint_path:
            logging.info("Loading parameters for the my CNN architectures...")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.error("there is no model file in model checkpoint paht.....")

    def predict(self, data_dict):

        data_df = pd.DataFrame.from_dict(data_dict, orient='index').T
        actual_batch_size = len(data_df)
        batch_X = []
        batch_idx = []
        for i in range(actual_batch_size):
            sample = data_df.iloc[i,:]
            array,idx = one_hot_representation(sample, self.fields_predict_dict, self.config['feature_length'])
            batch_X.append(array)
            batch_idx.append(idx)

        batch_X = np.array(batch_X)
        batch_idx = np.array(batch_idx)

        # create a feed dictionary for this batch
        feed_dict = {self.model.X: batch_X,
                     self.model.keep_prob:1,
                     self.model.feature_index : batch_idx}

        # shape of [None,2]
        y_out_prob = self.session.run([self.model.y_out_prob], feed_dict=feed_dict)
        return  y_out_prob[0][:, -1]

    def batch_predict(self, print_every = 50):
        """training model"""
        # get testing data, iterable
        test_data = pd.read_csv(Root_Dir + 'test.csv',
                                chunksize=self.model.batch_size)
        test_step = 1
        # batch_size data
        for data in test_data:
            actual_batch_size = len(data)
            batch_X = []
            batch_idx = []
            for i in range(actual_batch_size):
                sample = data.iloc[i,:]
                array,idx = one_hot_representation(sample, self.fields_predict_dict, self.config['feature_length'])
                batch_X.append(array)
                batch_idx.append(idx)

            batch_X = np.array(batch_X)
            batch_idx = np.array(batch_idx)

            # create a feed dictionary for this batch
            feed_dict = {self.model.X: batch_X,
                        self.model.keep_prob:1,
                        self.model.feature_index:batch_idx}

            # shape of [None,2]
            y_out_prob = self.session.run([self.model.y_out_prob], feed_dict=feed_dict)
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