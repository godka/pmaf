import numpy as np
import tensorflow as tf
import tflearn
import h5py

EPS = 1e-6
# https://arxiv.org/pdf/1406.2661.pdf
#disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
#gen_loss = -tf.reduce_mean(tf.log(disc_fake))


class DiscNetwork(object):

    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.batch_size = 64
        # initalized only in the optimizing process.
        self.generate_sample()

        self.fake_inputs = tflearn.input_data(
            shape=[None, self.s_dim[0], self.s_dim[1]])
        self.real_inputs = tflearn.input_data(
            shape=[None, self.s_dim[0], self.s_dim[1]])
        # Create the actor network
        self.fake_out = self.create_disc_network(self.fake_inputs)
        self.real_out = self.create_disc_network(self.real_inputs, True)

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
        self.optimize = tf.train.RMSPropOptimizer(
            self.lr_rate).minimize(self.obj)

    def create_disc_network(self, inputs, use=False):
        with tf.variable_scope('disc', reuse=use):
            # vmaf-score
            split_0 = tflearn.conv_1d(
                inputs[:, 0:1, :], 128, 4, activation='relu')
            # rebuffering
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, :-1], 128, activation='relu')
            # MOS on demand
            split_2 = tflearn.fully_connected(
                inputs[:, 2:3, -1], 128, activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            #split_1_flat = tflearn.flatten(split_1)

            merge_net = tflearn.merge(
                [split_0_flat, split_1, split_2], 'concat')

            dense_net_0 = tflearn.fully_connected(
                merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='sigmoid')

            return out

    def train(self, inputs):
        # Run fake & real
        # use trick: k>1
        sample_size = int(self.batch_size // 2)
        inputs = np.array(tflearn.data_utils.shuffle(inputs))[0]
        self.real_sample_pool = np.array(
            tflearn.data_utils.shuffle(self.real_sample_pool))[0]
        # for _ in range(self.k):
        _index = 0
        while(_index < inputs.shape[0]):
            self.sess.run(self.optimize, feed_dict={
                self.fake_inputs: inputs[_index: _index + sample_size],
                self.real_inputs: self.sample(sample_size)
            })
            _index += sample_size

    def generate_sample(self):
        print('generating real data...')
        f = h5py.File('baseline.hdf5', 'r')
        self.real_sample_pool = np.array(f['real'])
        f.close()

    def sample(self, sample_size):
        # let's rock&roll
        # picks sample from real sample pool
        i = np.random.randint(0, self.real_sample_pool.shape[0] - sample_size)
        return self.real_sample_pool[i:i+sample_size]

    def predict(self, inputs):
        # Run fake
        return self.sess.run(self.fake_out, feed_dict={
            self.fake_inputs: inputs
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
