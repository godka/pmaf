import tensorflow as tf
import tflearn
import numpy as np
import h5py
import load_trace
import fixed_env as env
import a3c
RANDOM_SEED = 42
A_DIM = 6
TEST_TRACES = './cooked_traces/'
S_INFO = 7  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
DEFAULT_QUALITY = 1  # default video quality without agent
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

class pretrain(object):

    def fromdata(self):
        _f = h5py.File(self.filename, 'r')
        X = np.array(_f['realx'])
        Y = np.array(_f['realy'])
        V = np.array(_f['realv'])
        _f.close()
        return X, Y, V

    def __init__(self, sess, actor, critic, rew, filename='train.h5'):
        self.sess = sess
        self.lr_rate = 1e-4
        self.actor = actor
        self.critic = critic
        self.disc = rew
        self.filename = filename
        self.trainX, self.trainY, self.trainV = self.fromdata()
        self.len = self.trainX.shape[0]
        self.s_dim = self.trainX.shape[1:]
        self.batch_size = 128
        #self.x_ = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
        self.action_ground = tflearn.input_data(shape=[None, A_DIM])
        self.action_loss = tflearn.objectives.categorical_crossentropy(
            self.actor.out, self.action_ground)
        self.action_optimize = tf.train.AdamOptimizer(
            self.lr_rate).minimize(self.action_loss)
        
        self.critic_ground = tflearn.input_data(shape=[None, 1])
        self.critic_loss = tflearn.objectives.mean_square(
            self.critic.out, self.critic_ground)
        self.critic_optimize = tf.train.AdamOptimizer(
            self.lr_rate).minimize(self.critic_loss)


    def train(self, epoch=15):
        self.train_generator(epoch=epoch)
        self.train_disc(epoch=epoch)
        self.train_generator_v(epoch=epoch)


    def train_generator(self, epoch=15):
        for i in range(epoch):
            # for i in self.
            _len = self.len
            idx = 0
            while(_len > 0):
                _tmp_len = np.minimum(self.batch_size, _len)
                _train_x = self.trainX[idx: idx+_tmp_len]
                _train_y = self.trainY[idx: idx+_tmp_len]
                self.sess.run(self.action_optimize, feed_dict={
                    self.actor.inputs: _train_x,
                    self.action_ground: _train_y})
                _len -= _tmp_len
                idx += _tmp_len
            print('[Pretrain] Generator:' + str(i))

    def train_disc(self, epoch=15):
        np.random.seed(RANDOM_SEED)
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw)
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, next_video_chunk_vmaf, \
            end_of_video, video_chunk_remain, video_chunk_vmaf = \
                net_env.get_video_chunk(bit_rate)
                
            if last_chunk_vmaf is None:
                last_chunk_vmaf = video_chunk_vmaf

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            last_bit_rate = bit_rate
            last_chunk_vmaf = video_chunk_vmaf

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = video_chunk_vmaf / 100.
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, :A_DIM] = np.array(next_video_chunk_vmaf) / 100.  # mega byte
            state[6, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            s_batch.append(state)
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_chunk_vmaf = None
                #del s_batch[:]
                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                s_batch.append(np.zeros((S_INFO, S_LEN)))
                video_count += 1

                if video_count >= len(all_file_names):
                    break
        v_s_batch = np.stack(s_batch, axis=0)
        for i in range(epoch):
            self.disc.train(v_s_batch)
            print('[Pretrain] Discriminator:' + str(i))

    def train_generator_v(self, epoch=15):
        np.random.seed(RANDOM_SEED)
        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw)
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        r_batch = []
        S_batch = []
        V_batch = []
        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, next_video_chunk_vmaf, \
            end_of_video, video_chunk_remain, video_chunk_vmaf = \
                net_env.get_video_chunk(bit_rate)
                
            if last_chunk_vmaf is None:
                last_chunk_vmaf = video_chunk_vmaf

            reward = self.disc.predict(s_batch[-1])
            reward = reward[0, 0]
            #d_batch.append(d_state)
            #trick,use gan loss.
            r_batch.append(np.log(reward))

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            last_bit_rate = bit_rate
            last_chunk_vmaf = video_chunk_vmaf

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = video_chunk_vmaf / 100.
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, :A_DIM] = np.array(next_video_chunk_vmaf) / 100.  # mega byte
            state[6, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = self.actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            s_batch.append(state)
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_chunk_vmaf = None
                
                #print(len(r_batch), len(v_batch))
                v_batch = a3c.discount(r_batch[:], 0.99)
                S_batch += s_batch[1:]
                V_batch += v_batch
                
                del s_batch[:]
                del r_batch[:]
                
                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                s_batch.append(np.zeros((S_INFO, S_LEN)))
                video_count += 1

                if video_count >= len(all_file_names):
                    break

        v_s_batch = np.stack(S_batch, axis=0)
        v_v_batch = np.vstack(V_batch)
        for i in range(epoch):
            # for i in self.
            _len = self.len
            idx = 0
            while(_len > 0):
                _tmp_len = np.minimum(self.batch_size, _len)
                _train_x = v_s_batch[idx: idx+_tmp_len]
                _train_y = v_v_batch[idx: idx+_tmp_len]
                self.sess.run(self.critic_optimize, feed_dict={
                    self.critic.inputs: _train_x,
                    self.critic_ground: _train_y})
                _len -= _tmp_len
                idx += _tmp_len
            print('[Pretrain] Critic:' + str(i))
