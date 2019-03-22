import tensorflow as tf
import tflearn
import a3c
import numpy as np
import env
import load_trace
import os
import h5py

S_INFO = 7
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
LOG_FILE = './results/log'
TRAIN_TRACES = './cooked_traces/'
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
RAND_RANGE = 1000
DEFAULT_QUALITY = 1  # default video quality without agent
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TRAIN_SEQ_LEN = 100
M_IN_K = 1000.0
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def copy_from_fake(real_y, fake_y):
    arr = []
    for (real, fake) in zip(real_y, fake_y):
        real_fake = np.array(fake, copy=True)
        np.random.shuffle(real_fake)
        real_action = np.argmax(real)
        _a = np.max(real_fake)
        _b = np.argmax(real_fake)
        tmp = real_fake[_a]
        real_fake[_a] = real_fake[_b]
        real_fake[_b] = tmp
        arr.append(real_fake)
    return np.vstack(arr)

def fromdata(filename):
    _f = h5py.File(filename, 'r')
    X = np.array(_f['realx'])
    Y = np.array(_f['realy'])
    V = np.array(_f['realv'])
    _f.close()
    return X, Y, V


def sample(real_state, real_action, sample_size=128):
    # let's rock&roll
    # picks sample from real sample pool
    i = np.random.randint(
        0, real_state.shape[0] - sample_size)
    return real_state[i:i+sample_size], real_action[i:i+sample_size]


def main():

    with tf.Session() as sess, open(LOG_FILE + '_train', 'w') as log_file:
        actor_inputs, actor_outputs = create_actor_network(
            [S_INFO, S_LEN], A_DIM)
        fake_disc = create_disc_network(actor_inputs, actor_outputs, False)
        real_inputs_state = tflearn.input_data(shape=[None, S_INFO, S_LEN])
        real_inputs_action = tflearn.input_data(shape=[None, A_DIM])
        real_disc = create_disc_network(
            real_inputs_state, real_inputs_action, True)

        gen_loss = - tf.reduce_mean(tf.log(fake_disc))

        disc_loss = - \
            tf.reduce_mean(tf.log(real_disc)) - \
            tf.reduce_mean(tf.log(1. - fake_disc))

        actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        disc_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')

        gen_optimize = tf.train.AdamOptimizer(
            ACTOR_LR_RATE).minimize(gen_loss, var_list=actor_vars)
        disc_optimize = tf.train.AdamOptimizer(
            CRITIC_LR_RATE).minimize(disc_loss, var_list=disc_vars)

        sess.run(tf.global_variables_initializer())
        # serve forever

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        #d_batch = []
        entropy_record = []
        time_stamp = 0
        chunk_index = 0

        real_x, real_y, real_v = fromdata('train.h5')

        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=42)

        while True:  # experience video streaming forever
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, next_video_chunk_vmaf, \
                end_of_video, video_chunk_remain, video_chunk_vmaf = \
                net_env.get_video_chunk(bit_rate)
            chunk_index += 1
            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            if last_chunk_vmaf is None:
                last_chunk_vmaf = video_chunk_vmaf

            reward = 0.8469011 * video_chunk_vmaf - 28.79591348 * rebuf + 0.29797156 * \
                np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - 1.06099887 * \
                np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - \
                2.661618558192494

            last_bit_rate = bit_rate
            last_chunk_vmaf = video_chunk_vmaf

            state = np.array(s_batch[-1], copy=True)
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[0, -1] = video_chunk_vmaf / 100.
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, :A_DIM] = np.array(
                next_video_chunk_vmaf) / 100.  # mega byte
            state[6, -1] = np.minimum(video_chunk_remain,
                                      CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            action_prob = sess.run(actor_outputs, feed_dict={
                actor_inputs: np.reshape(state, (1, S_INFO, S_LEN))
            })
            #actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(
                1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                # here we go
                state_batch = np.stack(s_batch, axis=0)
                action_batch = np.vstack(a_batch)
                x_batch, y_batch = sample(
                    real_x, real_y, sample_size=state_batch.shape[0])
                y_batch_ = copy_from_fake(y_batch, action_batch)
                sess.run(disc_optimize, feed_dict={
                    actor_inputs: state_batch,
                    real_inputs_state: x_batch,
                    real_inputs_action: y_batch_
                    # inputs_d_fake: d_fake
                })
                for _ in range(1):
                    sess.run(gen_optimize, feed_dict={
                        actor_inputs: state_batch
                        # real_inputs_state: x_batch,
                        #real_inputs_action: y_batch
                        # inputs_d_fake: d_fake
                    })

                log_file.write('\n')

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_chunk_vmaf = None
                chunk_index = 0

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)
                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def create_disc_network(state, action, isreuse=True):
    with tf.variable_scope('disc', reuse=isreuse):
        inputs = state
        split_0 = tflearn.fully_connected(
            inputs[:, 0:1, -1], 128, activation='relu')
        split_1 = tflearn.fully_connected(
            inputs[:, 1:2, -1], 128, activation='relu')
        split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
        split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
        split_4 = tflearn.conv_1d(
            inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
        split_5 = tflearn.conv_1d(
            inputs[:, 5:6, :A_DIM], 128, 4, activation='relu')
        split_6 = tflearn.fully_connected(
            inputs[:, 6:7, -1], 128, activation='relu')

        split_2_flat = tflearn.flatten(split_2)
        split_3_flat = tflearn.flatten(split_3)
        split_4_flat = tflearn.flatten(split_4)
        split_5_flat = tflearn.flatten(split_5)

        merge_net = tflearn.merge(
            [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5_flat, split_6], 'concat')

        dense_net_0 = tflearn.fully_connected(
            merge_net, 128, activation='relu')

        action_split = tflearn.fully_connected(
            action, 64, activation='relu')

        merge = tflearn.merge([dense_net_0, action_split], 'concat')
        merge = tflearn.fully_connected(
            merge, 64, activation='relu')
        merge = tflearn.fully_connected(
            merge, 1, activation='sigmoid')

        return merge


def create_actor_network(s_dim, a_dim):
    with tf.variable_scope('actor'):
        inputs = tflearn.input_data(shape=[None, s_dim[0], s_dim[1]])

        split_0 = tflearn.fully_connected(
            inputs[:, 0:1, -1], 128, activation='relu')
        split_1 = tflearn.fully_connected(
            inputs[:, 1:2, -1], 128, activation='relu')
        split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
        split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
        split_4 = tflearn.conv_1d(
            inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
        split_5 = tflearn.conv_1d(
            inputs[:, 5:6, :A_DIM], 128, 4, activation='relu')
        split_6 = tflearn.fully_connected(
            inputs[:, 6:7, -1], 128, activation='relu')

        split_2_flat = tflearn.flatten(split_2)
        split_3_flat = tflearn.flatten(split_3)
        split_4_flat = tflearn.flatten(split_4)
        split_5_flat = tflearn.flatten(split_5)

        merge_net = tflearn.merge(
            [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5_flat, split_6], 'concat')

        dense_net_0 = tflearn.fully_connected(
            merge_net, 128, activation='relu')
        out = tflearn.fully_connected(dense_net_0, a_dim, activation='softmax')

        return inputs, out


if __name__ == "__main__":
    main()
