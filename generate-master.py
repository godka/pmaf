import os
import sys
import numpy as np
import load_trace
import fixed_env as env
import h5py

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 7
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# [230,477,991,2056,5027,6000]
# [230,477,1427,2056,2962,6000]
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
# [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './cooked_test_traces/'
COMYCO_TRACES = './master/'

def get_chunk(bitrate):
    for i, p in enumerate(VIDEO_BIT_RATE):
        if p - bitrate == 0:
            return i

def read_logs(log_file):
    _file = open(COMYCO_TRACES + 'log_sim_vptpc_' + log_file, 'r')
    buffer = []
    for _lines in _file:
        #if len(_lines.split()) >= 2:
        _sp_lines = _lines.split()
        buffer.append(get_chunk(float(_sp_lines[1])))
    #buffer.reverse()
    _file.close()
    return np.array(buffer)


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(
        TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')
    action_buffer = read_logs(all_file_names[net_env.trace_idx])
    action_index = 0

    time_stamp = 0.

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = []
    a_batch = []
    #r_batch = []
    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        action_index += 1
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, next_video_chunk_vmaf, \
            end_of_video, video_chunk_remain, video_chunk_vmaf = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        # r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = np.zeros((S_INFO, S_LEN))
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
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

        s_batch.append(state)
        if action_index < action_buffer.shape[0]:
            #print(action_index, action_buffer.shape[0])
            bit_rate = action_buffer[action_index]

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1
        a_batch.append(action_vec)
        # entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            #miscast
            #s_batch.append(np.zeros((S_INFO, S_LEN)))
            #a_batch.append(action_vec)
            #entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')
            action_buffer = read_logs(all_file_names[net_env.trace_idx])
            action_index = 0

    print(len(s_batch), len(a_batch))
    f = h5py.File('train.h5', 'w')
    f['realx'] = np.array(s_batch)
    f['realy'] = np.array(a_batch)
    f.close()


if __name__ == '__main__':
    main()
