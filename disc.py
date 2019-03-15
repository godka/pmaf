import numpy as np
import tensorflow as tf
import tflearn
import h5py

EPS = 1e-6
# https://arxiv.org/pdf/1406.2661.pdf
#disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
#gen_loss = -tf.reduce_mean(tf.log(disc_fake))
A_DIM = 6


class DiscNetwork(object):

    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.batch_size = 128
        # initalized only in the optimizing process.
        self.real_sample_inputs, self.real_sample_actions = self.generate_sample()

        self.fake_inputs = tflearn.input_data(
            shape=[None, self.s_dim[0], self.s_dim[1]], name='fakeinput')
        self.real_inputs = tflearn.input_data(
            shape=[None, self.s_dim[0], self.s_dim[1]], name='realinput')
        self.fake_actions = tflearn.input_data(
            shape=[None, A_DIM], name='fakeinput')
        self.real_actions = tflearn.input_data(
            shape=[None, A_DIM], name='realinput')
        # Create the actor network
        self.fake_out = self.create_disc_network(
            self.fake_inputs, self.fake_actions)
        self.real_out = self.create_disc_network(
            self.real_inputs, self.real_actions, True)

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        # Compute loss functions
        self.obj = -tf.reduce_mean(tf.log(self.real_out + EPS)) - \
            tf.reduce_mean(tf.log(1. - self.fake_out + EPS))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(
            self.lr_rate).minimize(self.obj)

    def create_disc_network(self, inputs, actions, use=False):
        with tf.variable_scope('disc', reuse=use):
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], 128, activation='leaky_relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], 128, activation='leaky_relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], 128, 4, activation='leaky_relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], 128, 4, activation='leaky_relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :A_DIM], 128, 4, activation='leaky_relu')
            split_5 = tflearn.conv_1d(
                inputs[:, 5:6, :A_DIM], 128, 4, activation='leaky_relu')
            split_6 = tflearn.fully_connected(
                inputs[:, 6:7, -1], 128, activation='leaky_relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_5_flat = tflearn.flatten(split_5)

            merge_net = tflearn.merge([split_0, split_1, split_2_flat,
                                       split_3_flat, split_4_flat, split_5_flat, split_6], 'concat')

            dense_net_0 = tflearn.fully_connected(
                merge_net, 128, activation='relu')
            split_7 = tflearn.fully_connected(
                actions, 128, activation='leaky_relu')
            merge_net_1 = tflearn.merge([dense_net_0, split_7], 'concat')
            dense_net_0 = tflearn.fully_connected(
                merge_net_1, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='sigmoid')

            return out

    def train(self, inputs, actions):
        # Run fake & real
        # use trick: k>1
        sample_size = int(self.batch_size // 2)
        inputs, actions = tflearn.data_utils.shuffle(inputs, actions)
        self.real_sample_inputs, self.real_sample_actions = tflearn.data_utils.shuffle(
            self.real_sample_inputs, self.real_sample_actions)
        sample_inputs, sample_actions = self.sample(sample_size)
        # for _ in range(self.k):
        _index = 0
        while(_index < inputs.shape[0]):
            self.sess.run(self.optimize, feed_dict={
                self.fake_inputs: inputs[_index: _index + sample_size],
                self.fake_actions: actions[_index: _index + sample_size],
                self.real_inputs: sample_inputs,
                self.real_actions: sample_actions
            })
            _index += sample_size

    def generate_sample(self):
        #print('generating real data...')
        f = h5py.File('train.h5', 'r')
        real_sample_inputs = np.array(f['realx'])
        real_sample_actions = np.array(f['realy'])
        f.close()
        return real_sample_inputs, real_sample_actions

    def sample(self, sample_size):
        # let's rock&roll
        # picks sample from real sample pool
        i = np.random.randint(
            0, self.real_sample_inputs.shape[0] - sample_size)
        return self.real_sample_inputs[i:i+sample_size], self.real_sample_actions[i:i+sample_size]

    def predict(self, inputs, actions):
        # Run fake
        return self.sess.run(self.fake_out, feed_dict={
            self.fake_inputs: np.reshape(inputs, (-1, self.s_dim[0], self.s_dim[1])),
            self.fake_actions: np.reshape(actions, (-1, A_DIM))
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
