import tensorflow as tf
import tflearn
import numpy as np
import a3c
import h5py
import os
import disc
S_INFO = 7
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    #os.system('rm -rf /tmp/tflearn_logs')
    _f = h5py.File('train.h5', 'r')
    X = _f['realx']
    Y = _f['realy']
    V = _f['realv']
    x_len = int(X.shape[0] * 0.8)
    trainX = X[:x_len]
    testX = X[x_len:]
    trainY = Y[:x_len]
    testY = Y[x_len:]
    trainV = V[:x_len]
    testV = V[x_len:]
    (trainX, trainY, trainV) = tflearn.data_utils.shuffle(trainX, trainY, trainV)
    # Building convolutional network
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
        rew = disc.DiscNetwork(
            sess, state_dim=[S_INFO, S_LEN], learning_rate=ACTOR_LR_RATE / 10.)
        #predictions = actor.out
        #sess.run(tf.initialize_all_variables())
        trainer = tf.train.Saver()
        trainer.restore(sess, "models/pretrain.ckpt")
        # Train Generate Network
        # network = tflearn.regression(actor.out, optimizer='adam', learning_rate=1e-4,
        #                              loss='categorical_crossentropy', name='target')
        # model = tflearn.DNN(network, tensorboard_verbose=0)
        # model.fit({actor.inputs: trainX}, {'target': trainY}, n_epoch=15, 
        #         batch_size=128,
        #         validation_set=({actor.inputs: testX}, {'target': testY}),
        #         show_metric=True, shuffle=True, run_id='comyco2')
        # # Train Disc Network
        network2 = tflearn.regression(critic.out, optimizer='adam', learning_rate=1e-4,
                                     loss='mean_square', name='target2')
        model = tflearn.DNN(network2, tensorboard_verbose=0)
        model.fit({critic.inputs: trainX}, {'target2': trainV}, n_epoch=15, 
                batch_size=128,
                validation_set=({critic.inputs: testX}, {'target2': testV}),
                show_metric=True, shuffle=True, run_id='comyco3')

        model.save("models/pretrain.ckpt")

if __name__ == '__main__':
    main()
