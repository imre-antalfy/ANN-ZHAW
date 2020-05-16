# -*- coding: utf-8 -*-

#%% imports

# dependencies
from deepmusic.moduleloader import ModuleLoader
# predict next key
from deepmusic.keyboardcell import KeyboardCell
# encapsulate song data, to run get_scale, get_relative_methods
import deepmusic.songstruct as music
# numpy
import numpy as np
import tensorflow as tf

import tensorflow_addons

#%% build network
def build_network(self):
    # create computation graph, encapsulate session and the graph init
    # session creation for tensorflow
    input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()
    
    
    # define placeholer for inputs
    # this will be for the notes
    with tf.name('placeholder_inputs'):
        self.inputs = [
            tf.placeholder(
                tf.float32, #numerical data for midi
                [self.args.batch_size, input_dim], # how much data will be fed?
                name='input' 
                )
            ]

    # targets 88 key, binary classification problem
    with tf.name_scope('placeholder_targets'):
        self.targets = [
            tf.placeholder(
                tf.int32 # input is either 1 or 0
                [self.batch_size],
                name='target'
                )
            ]
        
    # as its an RNN, feed in the porevious hidden state
    with tf.name_scope('placeholder_use_prev'):
        self.use_prev = [
            tf.placeholder(
                tf.bool,
                [],
                name='use_prev')            
            ]
    
    # define network
    self.loop_processing = ModuleLoader.loop_processing.build_module(self.args)
    def loop_rnn(prev, i):
        next_input = self.loop_processing(prev)
        return tf.cond(self.prev[i], lambda: next_input, lambda: self.inputs[i])
    
    # build sequence to sequence model
    self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder( #!!! didft find seq2seq
        decoder_inputs = self.inputs,
        initial_state = None,
        cell = KeyboardCell, # because defined in keyboard cell
        loop_function = loop_rnn
    )
    
    # as mutliple notes can be pressed, its a multiclass prob
    # use cross entropy as loss function
    # measuring the difference of two prob distribs
    
    loss_fct = tf.nn.seq2seq.sequence_loss(
        self.outputs,
        self.targets,
        softmax_loss_function = tf.nn.softmax.cross_entropy_with_logits,
        average_accross_timesteps = True,
        average_accross_batch = True    
        )
    
    # initialize optimizer
    opt = tf.train.AdamOptimizer(
        learning_rate = self.current_learning_rate,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-08 )

    self.opt.op = opt.minimize(loss_fct)    
    
    
    
    
    
    
    
    
    
    
    