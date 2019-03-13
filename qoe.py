import tensorflow as tf
import tflearn
import numpy as np

class qoe(object):
    def __init__(self, sess):
        self.sess = sess
        self.inputs, self.out = self.create_network()
        trainer = tf.train.Saver()
        trainer.restore(self.sess, 'qoe/model.ckpt')
        #y_ = sess.run(net, feed_dict={inputs: X})
        #alpha_ = sess.run(soft, feed_dict={inputs: X})

    def conv_1d_res_block(self, net, filter_num, filter_size):
        _shape = net.get_shape().as_list()
        _filter_n = _shape[-1]
        #fc_net = tflearn.fully_connected(net, filter_num, activation='relu')
        cnn_net = tflearn.conv_1d(net, filter_num, 1, activation='relu')
        cnn_net = tflearn.batch_normalization(cnn_net)
        cnn_net = tflearn.conv_1d(
            cnn_net, filter_num, filter_size, activation='relu')
        cnn_net = tflearn.batch_normalization(cnn_net)
        cnn_net = tflearn.conv_1d(cnn_net, _filter_n, 1, activation='relu')
        cnn_net = tflearn.batch_normalization(cnn_net)
        # cnn_net: 1,3,64
        out = tf.add(net, cnn_net)
        return out

    def create_network(self):
        inputs = tflearn.input_data(shape=[None, 3], name='input')
        softnet = tf.expand_dims(inputs, -1)
        softnet = tflearn.conv_1d(softnet, 64, 1, activation='relu')
        softnet = tflearn.batch_normalization(softnet)
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

        return inputs, net

    def predict(self, inputs):
        y_ = self.sess.run(self.out, feed_dict={
                      self.inputs: np.reshape(inputs, (-1, 3))})
        return y_[0][0]
