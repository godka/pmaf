import tensorflow as tf
import tflearn
import numpy as np
import h5py

class qoe(object):
    def __init__(self, sess, learning_rate):
        self.sess = sess
        self.inputs, self.out = self.create_network()
        self.lr_rate = learning_rate
        self.outputs = tflearn.input_data(shape=[None, 1], name='output')
        f = h5py.File("qoe.hdf5", "r")
        self.x = np.array(f['x'])
        self.y_ = np.array(f['y'])
        f.close()
        # Compute loss functions
        self.obj = tflearn.objectives.mean_square(self.out, self.outputs)
        self.batch_size = 64
        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(
            self.lr_rate).minimize(self.obj)
        #trainer = tf.train.Saver()
        #trainer.restore(self.sess, 'qoe/model.ckpt')
        #y_ = sess.run(net, feed_dict={inputs: X})
        #alpha_ = sess.run(soft, feed_dict={inputs: X})
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qoe')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

    def conv_1d_res_block(self, net, filter_num, filter_size):
        _shape = net.get_shape().as_list()
        _filter_n = _shape[-1]
        #fc_net = tflearn.fully_connected(net, filter_num, activation='relu')
        cnn_net = tflearn.conv_1d(net, filter_num, 1, activation='relu')
        #cnn_net = tflearn.batch_normalization(cnn_net)
        cnn_net = tflearn.conv_1d(
            cnn_net, filter_num, filter_size, activation='relu')
        #cnn_net = tflearn.batch_normalization(cnn_net)
        cnn_net = tflearn.conv_1d(cnn_net, _filter_n, 1, activation='relu')
        #cnn_net = tflearn.batch_normalization(cnn_net)
        # cnn_net: 1,3,64
        out = tf.add(net, cnn_net)
        return out

    def create_network(self):
        with tf.variable_scope('qoe'):
            inputs = tflearn.input_data(shape=[None, 3], name='input')
            softnet = tf.expand_dims(inputs, -1)
            softnet = tflearn.conv_1d(softnet, 64, 1, activation='relu')
            #softnet = tflearn.batch_normalization(softnet)
            for p in range(20):
                softnet = self.conv_1d_res_block(softnet, 64, 3)
            #softnet = tf.expand_dims(softnet, 1)
            #soft = tflearn.global_avg_pool(softnet)
            soft = tflearn.fully_connected(softnet, 64, activation='relu')
            soft = tflearn.fully_connected(soft, 2, activation='linear')
            soft = tf.expand_dims(soft, axis=-1)

            ret = tf.expand_dims(inputs[:, 1:], 1)
            net = tf.matmul(ret, soft)
            net = tf.reshape(net, (-1, 1))
            net = tf.add(net, inputs[:, 0:1])
            net = tf.nn.sigmoid(net)
            return inputs, net

    def predict(self, inputs):
        y_ = self.sess.run(self.out, feed_dict={
                      self.inputs: np.reshape(inputs, (-1, 3))})
        return y_

    def train(self):
        self.x, self.y_ = tflearn.data_utils.shuffle(self.x, self.y_)
        self.sess.run(self.optimize, feed_dict={
            self.inputs: self.x[:self.batch_size],
            self.outputs: self.y_[:self.batch_size]
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
