from __future__ import absolute_import
from __future__ import division

import math
import os
import random
import sys
import time
import random

import numpy as np
import tensorflow as tf

import exps.nlc_env
from exps.nlc_env import pair_iter, NLCEnv

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 40, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/nlc_data/",
                           "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./tmp", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")

FLAGS = tf.app.flags.FLAGS



if __name__ == '__main__':
    # ======= Test on Env =========
    env = NLCEnv()
    env.reset()
    print env.observation_space
    print env.action_space
