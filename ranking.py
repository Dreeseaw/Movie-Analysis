'''
William Dreese
Movie Dataset Analysis
ranking.py

uses a binary MLP to find relations between genres
that lead to highest box office revenue
'''

from __future__ import print_function

import datetime
import readnclean as rc

import numpy as np
import pandas as pd
import tensorflow as tf

from operator import itemgetter

def get_genres(file_name):
    h = open(file_name,"r")
    genres = (h.readline().split(","))[:-1]
    h.close()
    return genres

''' returns top n genres '''
def get_top(n, rnk, genres):
    top = [-1]*n
    for r in range(n):
        maxval = -100
        maxind = -1
        for v in range(len(genres)):
            if rnk[0][v][0] > maxval and v not in top:
                maxind = v
                maxval = rnk[0][v][0]
        top[r] = maxind
    return [genres[t] for t in top]
            

def run(train_file, hidden_layer_size=64, act_function="relu", layers=2, gpu=False, opti="sgd", var_init="normal"):

    config = tf.ConfigProto(allow_soft_placement = True)
    ret_top = list()

    gorc = "/device:gpu:0"
    if not gpu: gorc = "/device:cpu:0" 

    # meta-parameters
    learning_rate = 0.001
    max_training_epochs = 200
    annel_rate = 50
    batch_size = 20

    ''' I left lots of extra code laying around from when I was still
        developing the model, in case you wanna mess around with it '''
    def lecun_tanh(x): return 1.7159 * tf.nn.tanh(2.0*x/3.0)
    act_func = lecun_tanh
    if act_function == "tanh": act_func = tf.nn.tanh
    if act_function == "relu": act_func = tf.nn.relu
    if act_function == "leaky_relu": act_func = tf.nn.leaky_relu
    if act_function == "relu6": act_func = tf.nn.relu6

    n_hidden_1 = hidden_layer_size
    n_hidden_2 = hidden_layer_size
    n_hidden_3 = hidden_layer_size
    n_classes = 1

    ''' data (out) '''
    write_file = "ranking_model_output.txt"
    f = open(write_file,"w+") 

    ''' data (in) '''
    train_size = 3400
    genres = list()

    train_file += ".csv"
    genres = get_genres(train_file)
    n_input = len(genres)
    dataset = tf.data.TextLineDataset(train_file).skip(1)
    
    def parse_csv(line):
        cols_types = [[]] * (n_input+1)
        columns = tf.decode_csv(line, record_defaults=cols_types,field_delim=',')
        return tf.stack(columns)
    
    dataset = dataset.batch(batch_size).map(parse_csv)
    evalset = np.diag(np.ones((n_input)))

    ''' setting gpu to true will run on gpu '''
    with tf.device(gorc):
        
        X = tf.placeholder(tf.float32,[None, n_input])
        Y = tf.placeholder(tf.float32,[None, n_classes])

        def var_inits(s):
            if var_init == "glorut":
                ini = tf.contrib.layers.xavier_initializer()
                return tf.Variable(ini(s),dtype=tf.float32)
            return tf.Variable(tf.random_normal(shape=s, stddev=0.001,dtype=tf.float32,seed=1998))
            
        weights = {
            'h1':  var_inits([n_input, n_hidden_1]),
            'h2':  var_inits([n_hidden_1, n_hidden_2]),
            'h3':  var_inits([n_hidden_2, n_hidden_3]),
            'out': var_inits([n_hidden_3, n_classes])
        }
        biases = {
            'b1':  var_inits([n_hidden_1]),
            'b2':  var_inits([n_hidden_2]),
            'b3':  var_inits([n_hidden_3]),
            'out': var_inits([n_classes])
        }

        ''' nothing too intense here, just a MLP '''
        def multilayer_perceptron(x):
            layer_1   = act_func(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
            layer_2   = act_func(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
            layer_3   = act_func(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
            if layers == 2: layer_3 = layer_1
            elif layers == 3: layer_3 = layer_2
            out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
            return out_layer

        ''' loss function, optimizers, accuracy op '''
        logits   = multilayer_perceptron(X)
        loss_op  = tf.abs(tf.reduce_sum(Y - logits))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
        
    print("\n-----model made----\n")

    with tf.Session() as sess:

        sess = tf.Session(config = config)

        #inits
        tf.set_random_seed(1998)
        sess.run(tf.global_variables_initializer())
        
        #epoch loop
        for ep_count in range(max_training_epochs):
            
            f.write("\nEpoch: "+str(ep_count+1))
            print("\nEpoch: "+str(ep_count+1))
            
            avg_loss = 0
            train_batch_iter = dataset.make_one_shot_iterator().get_next()            
            RUNTIME_START = datetime.datetime.now()

            ''' training loop '''
            for i in range(train_size//batch_size):

                ''' get a batch and use some quick pre-pro '''
                batch_wierd = sess.run(train_batch_iter)
                feed_x = batch_wierd[:-1].T
                feed_x /= feed_x.sum(axis=1,keepdims=True)[:]
                feed_y = np.reshape(batch_wierd[-1].T,(batch_size,1))
                
                _,loss = sess.run([train_op,loss_op], feed_dict={X: feed_x, Y: feed_y})
                avg_loss += np.sum(loss) / batch_size

            avg_loss /= train_size//batch_size
            
            f.write("\n\tTraining Loss: {:.4f}".format(avg_loss))
            print("\n\tTraining Loss: {:.4f}".format(avg_loss))

            ''' test all 82 combos, display top 5 '''
            rankings = sess.run([logits], feed_dict={X: evalset})
            top = get_top(10, rankings, genres)
            f.write("\n\tTop 10: "+str(top))
            print(top)
            ret_top = top
            
            RUNTIME_END = datetime.datetime.now()
            f.write("\n\tRuntime: " + str(RUNTIME_END - RUNTIME_START))
            print("\n\tRuntime: " + str(RUNTIME_END - RUNTIME_START))
        
        f.close()
        return ret_top[:5]
