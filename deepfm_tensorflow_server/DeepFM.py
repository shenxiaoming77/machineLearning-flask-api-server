# coding:utf-8
import os
import sys
import tensorflow as tf
import numpy as np

import math
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from  .settings import  *
from  .utilities import *

import  pickle

class DeepFM(object):
    """
    Deep FM with FTRL optimization
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']  #40
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']

        # num of features
        #self.p = feature_length  #number of one-hot coding features
        self.p = config['feature_length']

        # num of fields
        #self.field_cnt = field_cnt   #21
        self.field_cnt = config['field_cnt']

    def add_placeholders(self):
        self.X = tf.placeholder('float32', [None, self.p])
        self.y = tf.placeholder('int64', [None,])

        # index of none-zero features
        #下面的batch_idx 则是用于某一轮的batch 训练数据的idx 集合，大小为:batch_size * field_cnt
        # 用于inference接口里面DNN模块中抽取V隐向量矩阵的feature_index
        self.feature_index = tf.placeholder('int64', [None, self.field_cnt])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01),dtype='float32')

        # Factorization Machine
        with tf.variable_scope('FM'):
            b = tf.get_variable('bias_fm', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w_fm', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,tf.reduce_mean(
                                                     tf.subtract( tf.pow(tf.matmul(self.X, v), 2),
                                                                  tf.matmul(tf.pow(self.X, 2), tf.pow(v, 2))),
                                                     1, keep_dims=True))
            # shape of [None, 2]
            self.y_fm = tf.add(self.linear_terms, self.interaction_terms)

        # three-hidden-layer neural network, network shape of (200-200-200)
        #tf.gather: 按feature_index 来抽取子集，在axis=0维度上，按index 抽取若干行数据，行下标可以不连续
        #tf.reshape: 对矩阵进行变换
        #feature_index = tf.placeholder('int64', [None, self.field_cnt])
        #v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01),dtype='float32')

        '''
        data = [[[1, 1, 1], [2, 2, 2]],
         [[3, 3, 3], [4, 4, 4]],
         [[5, 5, 5], [6, 6, 6]]]

        index = [[0, 2],[1, 1]]
        gather_data = tf.gather(data, index)

        其中index的shape为[2, 2]
        index[0] = [0, 2], 表示抽取data中的第一行和第三行作为子集
        [[[1, 1, 1], [2, 2, 2]],
         [[5, 5, 5], [6, 6, 6]]]

        index[1] = [1, 1], 表示抽取data中的第二行和第二行作为子集
        [[[3, 3, 3], [4, 4, 4]],
         [[3, 3, 3], [4, 4, 4]]]
        '''

        #在每次迭代中，feature_index的shape实际为[batch_size, field_cnt]
        #feature_index中每一元素为idx数组，存储的时某个x样本的21个原始特征值对应的index，因此

        with tf.variable_scope('DNN',reuse=False):
            # embedding layer
            y_embedding_input = tf.reshape(tf.gather(v, self.feature_index), [-1, self.field_cnt * self.k])

            # first hidden layer
            w1 = tf.get_variable('w1_dnn', shape=[self.field_cnt * self.k, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b1 = tf.get_variable('b1_dnn', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l1 = tf.nn.relu(tf.matmul(y_embedding_input, w1) + b1)

            # second hidden layer
            w2 = tf.get_variable('w2_dnn', shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b2 = tf.get_variable('b2_dnn', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l2 = tf.nn.relu(tf.matmul(y_hidden_l1, w2) + b2)

            # third hidden layer
            w3 = tf.get_variable('w3_dnn', shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b3 = tf.get_variable('b3_dnn', shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l3 = tf.nn.relu(tf.matmul(y_hidden_l2, w3) + b3)

            # output layer
            w_out = tf.get_variable('w_out', shape=[200, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            b_out = tf.get_variable('b_out', shape=[2],
                                 initializer=tf.constant_initializer(0.001))
            self.y_dnn = tf.nn.relu(tf.matmul(y_hidden_l3, w_out) + b_out)

        # add FM output and DNN output
        self.y_out = tf.add(self.y_fm, self.y_dnn)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        # 在tf.argmax( , )中有两个参数，第一个参数是矩阵，
        # 第二个参数是0或者1。0表示的是按列比较返回最大值的索引，1表示按行比较返回最大值的索引
        # 所以tf.argmax 返回的是最大值的索引，这里self.out 为[None, 2]维矩阵，因此返回的是对每行的两个值比较，
        # 得到的最大值索引，0或者1， 以此来代表预测的类别是0或者1
        #self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out,1), tf.int64), model.y)
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out, 1), tf.int64), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.FtrlOptimizer(self.lr, #learningRate
                                           l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step = self.global_step)

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()



def train_model(sess, model, epochs=10, print_every=500):
    """training model"""
    num_samples = 0
    losses = []
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    for e in range(epochs):
        print('epoch: ', e)

        # get training data, iterable
        train_data = pd.read_csv(Root_Dir + 'train.csv',
                                 chunksize=model.batch_size)
        # batch_size data
        for data in train_data:
            actual_batch_size = len(data)
            batch_X = []
            batch_y = []
            batch_idx = []
            for i in range(actual_batch_size):
                sample = data.iloc[i,:]
                #array 是one-hot编码后的向量，长度为feature_size（one-hot编码特征的数量）
                #idx则是与array相对应的one-hot编码特征中所有非零特征的索引集合，长度为21，里面的值为这21个原始特征对应的索引值，
                # 大小范围为0 ~ (feature_size-1)
                #相当于对于一个长度为feature_size的初始零向量array_0，找到field_cnt个原始特征对应的index，对array_0中相应位置设置为1
                #这样就得到了一个one-hot编码的向量array

                #下面的batch_idx 则是用于某一轮的batch 训练数据的idx 集合，大小为:batch_size * field_cnt
                # 用于inference接口里面DNN模块中抽取V隐向量矩阵的feature_index

                '''
                a[-1]    # last item in the array
                a[-2:]   # last two items in the array
                a[:-2]   # everything except the last two items
                '''

                array, idx = one_hot_representation(sample,fields_train_dict, train_array_length)
                batch_X.append(array[:-2])
                batch_y.append(array[-1])
                batch_idx.append(idx)

            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            batch_idx = np.array(batch_idx)

            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_X,
                         model.y: batch_y,
                         model.feature_index: batch_idx,
                         model.keep_prob:1}

            loss, accuracy,  summary, global_step, _ = sess.run([model.loss,
                                                                 model.accuracy,
                                                                 merged,
                                                                 model.global_step,
                                                                 model.train_op],
                                                                 feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss * actual_batch_size)

            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            # print training loss and accuracy
            if global_step % print_every == 0:
                logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                             .format(global_step, loss, accuracy))
                saver.save(sess, "checkpoints/model", global_step = global_step)

        # print loss of one epoch
        total_loss = np.sum(losses) / num_samples
        print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e+1))

def validation_model(sess, model, print_every=50):
    """testing model"""
    # num samples
    num_samples = 0
    # num of correct predictions
    num_corrects = 0
    losses = []
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('test_logs', sess.graph)
    # get testing data, iterable
    validation_data = pd.read_csv(Root_Dir + 'train.csv',
                                  chunksize=model.batch_size)
    # testing step
    valid_step = 1
    # batch_size data
    for data in validation_data:
        actual_batch_size = len(data)
        batch_X = []
        batch_y = []
        batch_idx = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array,idx = one_hot_representation(sample,fields_train_dict, train_array_length)
            batch_X.append(array[:-2])
            batch_y.append(array[-1])
            batch_idx.append(idx)
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        batch_idx = np.array(batch_idx)
        # create a feed dictionary for this batch,
        feed_dict = {model.X: batch_X, model.y: batch_y,
                 model.feature_inds: batch_idx, model.keep_prob:1}
        loss, accuracy, correct, summary = sess.run([model.loss, model.accuracy,
                                                     model.correct_prediction, merged,],
                                                    feed_dict=feed_dict)
        # aggregate performance stats
        losses.append(loss*actual_batch_size)
        num_corrects += correct
        num_samples += actual_batch_size
        # Record summaries and train.csv-set accuracy
        test_writer.add_summary(summary, global_step=valid_step)
        # print training loss and accuracy
        if valid_step % print_every == 0:
            logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                         .format(valid_step, loss, accuracy))
        valid_step += 1
    # print loss and accuracy of one epoch
    total_correct = num_corrects/num_samples
    total_loss = np.sum(losses)/num_samples
    print("Overall test loss = {0:.3g} and accuracy of {1:.3g}" \
          .format(total_loss,total_correct))







if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # seting fields
    fields_train = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                    'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                    'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
                    'device_conn_type','click']

    # loading dicts
    fields_train_dict = {}
    for field in fields_train:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_train_dict[field] = pickle.load(f)

    # length of representation
    train_array_length = max(fields_train_dict['click'].values()) + 1

    # initialize the model
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 512
    config['reg_l1'] = 2e-3
    config['reg_l2'] = 0
    config['k'] = 40

    # get feature length
    feature_length = train_array_length - 2
    config['feature_length'] = feature_length

    print('feature length: ', feature_length)

    # num of fields
    field_cnt = 21
    config['field_cnt'] = field_cnt

    model = DeepFM(config)

    # build graph for model
    model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)


    with tf.Session() as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        print('start training...')
        train_model(sess, model, epochs=10, print_every=100)
        # print('start validation...')
        # validation_model(sess, model, print_every=100)
