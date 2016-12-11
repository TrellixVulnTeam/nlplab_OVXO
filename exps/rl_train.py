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
import calendar
import os.path as osp

from sandbox.rocky.tf.envs.base import TfEnv
from exps.nlc_env import NLCEnv, DataDistributor, levenshtein
# from exps.preprocess import pair_iter
from exps.nlc.util import pair_iter
from exps.nlc.nlc_model import NLCModel
import exps.nlc.nlc_data as nlc_data
from exps.policy import CategoricalGRUPolicy
from sandbox.rocky.tf.core.network import MLP, Baseline, GRUNetwork
# from exps.baseline_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.misc.overrides import overrides
from exps.nlc_env import build_data
from exps.network import RewardGRUNetwork
from exps.actor_critic import ActorCritic
from sandbox import RLLabRunner
from rllab.config import LOG_DIR
from tensorflow.python.platform.flags import _define_helper
import sandbox.rocky.tf.core.layers as L

"""
Assemble the end-to-end encoder-decoder
train normally
then put the model into policy construction
then train it with
"""

# tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
# tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
# tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
# tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")  # 40
# tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")  # 400
# tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")  # 3
# tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
# tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
# tf.app.flags.DEFINE_string("data_dir", "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/",
#                            "Data directory")
# tf.app.flags.DEFINE_string("train_dir", "./tmp", "Training directory.")
# tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
# tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
# tf.app.flags.DEFINE_string("baseline_type", 'mlp', "linear|mlp")

tf.app.flags.DEFINE_bool("rl_only", False, "flag True to only train rl portion")

tf.app.flags.DEFINE_string("exp_name", "actor_critic", "name unique experiment")
tf.app.flags.DEFINE_string('tabular_log_file', 'tab.txt', "")
tf.app.flags.DEFINE_string('text_log_file', 'text.txt', "")
tf.app.flags.DEFINE_string('params_log_file', 'args.txt', "")
tf.app.flags.DEFINE_string('snapshot_mode', 'all', "")
tf.app.flags.DEFINE_string('log_tabular_only', False, "")
tf.app.flags.DEFINE_string('log_dir', "./logs/", "")
_define_helper('args_data', None, "should be none", flagtype=object)

# goes through data once!
tf.app.flags.DEFINE_integer("n_itr", 1314, "this pulls sentences from data")

FLAGS = tf.app.flags.FLAGS


def create_model(session, vocab_size, forward_only):
    model = NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
        print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def get_tokenizer(FLAGS):
    tokenizer = nlc_data.char_tokenizer if FLAGS.tokenizer.lower() == 'char' else nlc_data.basic_tokenizer
    return tokenizer


def validate(model, sess, x_dev, y_dev):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_dev, y_dev, FLAGS.batch_size,
                                                                            FLAGS.num_layers):
        # [32, 64] (time step, batch_size)
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def get_cer(model, sess, x_train, y_train):
    from exps.nlc.decode import decode_beam
    rl_batchsize = 32  # matches the enviornment batch

    total_cer = []
    for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, rl_batchsize,
                                                                            FLAGS.num_layers):
        # we decode them and compute CER per batch, and do AVERAGE!
        # in terms of evaluation of the trained policy, we just take the last 50 iterations' rewards and
        # calculate mean
        encoder_output = model.encode(sess, source_tokens, source_mask)
        beam_toks, probs = decode_beam(model, sess, encoder_output, max_beam_size=2)  # beam-size fixed at 1
        # we only do maximum decoding, no beam decoding
        best_tok = beam_toks[0]  # first one is the best one
        print best_tok

        sys.exit(0)



def train_seq2seq(model, sess, x_dev, y_dev, x_train, y_train):
    print('Initial validation cost: %f' % validate(model, sess, x_dev, y_dev))

    if False:
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        print ("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

    epoch = 0
    previous_losses = []
    exp_cost = None
    exp_length = None
    exp_norm = None
    while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
        epoch += 1
        current_step = 0

        ## Train
        for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, FLAGS.batch_size,
                                                                                FLAGS.num_layers):
            # Get a batch and make a step.
            tic = time.time()

            grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

            toc = time.time()
            iter_time = toc - tic
            current_step += 1

            lengths = np.sum(target_mask, axis=0)
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)

            if not exp_cost:
                exp_cost = cost
                exp_length = mean_length
                exp_norm = grad_norm
            else:
                exp_cost = 0.99 * exp_cost + 0.01 * cost
                exp_length = 0.99 * exp_length + 0.01 * mean_length
                exp_norm = 0.99 * exp_norm + 0.01 * grad_norm

            cost = cost / mean_length

            if current_step % FLAGS.print_every == 0:
                print(
                    'epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, batch time %f, length mean/std %f/%f' %
                    (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, iter_time,
                     mean_length,
                     std_length))

        ## Checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        ## Validate
        valid_cost = validate(model, sess, x_dev, y_dev)

        print("Epoch %d Validation cost: %f" % (epoch, valid_cost))

        if len(previous_losses) > 2 and valid_cost > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
        previous_losses.append(valid_cost)
        sys.stdout.flush()

    return model


class BaselineMLP(MLP, Baseline):
    """
    Same as the old BaselineMLP with optimizer replaced
    with FirstOrderOptimizer()
    (Should be ok)

    Baseline must implement 2 methods:
    fit_with_samples()
    fit()

    Used in nlplab/rllab/sampler/base.py
    """

    def __init__(self, name, output_dim, hidden_sizes, hidden_nonlinearity, output_nonlinearity,
                 hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False,
                 weight_normalization=False):
        self.save_name = name
        super(BaselineMLP, self).__init__(name, output_dim, hidden_sizes, hidden_nonlinearity, output_nonlinearity,
                                          hidden_W_init, hidden_b_init, output_W_init, output_b_init, batch_size,
                                          input_var, input_layer, input_shape, batch_normalization,
                                          weight_normalization)

    def initialize_optimizer(self):
        self._optimizer = FirstOrderOptimizer()

        optimizer_args = dict(
            loss=self.loss(),
            target=self,
            inputs=[self.input_var, self.target_var],
            network_outputs=[self.output]
        )

        self._optimizer.update_opt(**optimizer_args)

    @overrides
    def predict(self, path):
        X = path['observations']
        return super(BaselineMLP, self).predict(X)

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        # self._regressor.fit(observations, returns.reshape((-1, 1)))
        self._optimizer.optimize([observations, returns[..., None]])


if __name__ == '__main__':

    # comment this out when running for real
    try:
        os.remove(FLAGS.data_dir + "weights/NLC_weights.h5")
    except:
        pass

    # ======== End2End Normal Training =========

    # Train a translation model using NLC data.
    # Prepare NLC data.
    print("Preparing NLC data in %s" % FLAGS.data_dir)

    # TODO:  FLAGS.tokenizer.lower() doesn't belong here (we are only doing char)
    # x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
    #     FLAGS.data_dir, FLAGS.max_vocab_size,
    #     tokenizer=get_tokenizer(FLAGS))  # FLAGS.tokenizer.lower()

    # path_2_ptb_data = "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/"
    path_2_ptb_data = FLAGS.data_dir + "/ptb_data"

    # x_train = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/train.ids.x"
    # y_train = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/train.ids.y"

    # x_dev = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/valid.ids.x"
    # y_dev = "/Users/Aimingnie/Documents" + "/School/Stanford/AA228/nlplab/ptb_data/valid.ids.y"

    x_train = "{}/train.ids.x".format(path_2_ptb_data)
    y_train = "{}/train.ids.y".format(path_2_ptb_data)

    x_dev = "{}/valid.ids.x".format(path_2_ptb_data)
    y_dev = "{}/valid.ids.y".format(path_2_ptb_data)

    vocab_path = "{}/vocab.dat".format(path_2_ptb_data)

    source_tokens, source_mask, target_tokens, target_mask = build_data(fnamex="{}/train.ids.x".format(path_2_ptb_data),
                                                                        fnamey="{}/train.ids.y".format(path_2_ptb_data),
                                                                        num_layers=1, max_seq_len=200)

    vocab, _ = nlc_data.initialize_vocabulary(vocab_path)

    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, vocab_size, False)

        if not FLAGS.rl_only:
            model = train_seq2seq(model, sess, x_dev, y_dev, x_train, y_train)

        get_cer(model, sess, x_train, y_train)

        # save weights to h5
        model.save_decoder_to_h5(sess, title="NLC_weights")

        # ======== Actor-critic Training =========
        # note that we are still inside the TF session scope

        # building an enviornment
        # notice we are still in previous session!
        # also notice that rllab will create a new session in Batchpolot

        L_dec = model.L_dec.eval(session=sess)
        L_enc = model.L_enc.eval(session=sess)

        config = {
            "gru_size": FLAGS.size,  # remember train.py for NLC must be adjusted as well
            "source": source_tokens,
            "target": target_tokens,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "model": model,
            "L_dec": L_dec,  # vocab_size: 52, hidden_size: 50
            "L_enc": L_enc,
            "measure": "CER",
            'data_dir': FLAGS.data_dir + "ptb_data/",
            "max_seq_len": 32,
            # This is determined by data preprocessing (for decoder it's 32, cause <start>, <end>)
            "batch_size": 1024,  # 32,  # batch_size must be multiples of max_seq_len (consider a MUCH BIGGER BATCH)
            # batch-size with n_env in policy code???)
            "sess": sess,  # tf_session
            "vocab_size": L_dec.shape[0],
            "encoder_max_seq_length": 30
        }
        ddist = DataDistributor(config)

        env = TfEnv(NLCEnv(ddist, config))

        # create a policy

        policy = CategoricalGRUPolicy(name="gru_policy", env_spec=env.spec,
                                      distributor=ddist, config=config,
                                      hidden_dim=config["gru_size"],
                                      hidden_nonlinearity=tf.nn.relu,
                                      word_embed_dim=config["gru_size"],
                                      )  # in character Seq2Seq,
        # word_emb and hid_size are the same
        policy.load_params("NLC_weights", 0, [])

        # create a delayed policy network
        delayed_policy = CategoricalGRUPolicy(name="gru_delayed_policy", env_spec=env.spec,
                                              distributor=ddist, config=config,
                                              hidden_dim=config["gru_size"],
                                              hidden_nonlinearity=tf.nn.relu,
                                              word_embed_dim=config["gru_size"],
                                              )

        delayed_policy.load_params("NLC_weights", 0, [])

        # create a critic network

        critic = RewardGRUNetwork(name="gru_critic", config=config,
                                  hidden_dim=config["gru_size"],
                                  hidden_nonlinearity=tf.nn.relu, )

        # create a target critic (assign same weights)

        target_critic = RewardGRUNetwork(name="gru_target_critic", config=config,
                                         hidden_dim=config["gru_size"],
                                         hidden_nonlinearity=tf.nn.relu, )

        # pretrain critic (let's not pretrain it yet)

        # create baseline (paper didn't have a baseline)
        baseline = BaselineMLP(name='mlp_baseline',
                               output_dim=1,
                               hidden_sizes=[config["gru_size"]],  # same size as GRU..can be changed
                               hidden_nonlinearity=tf.nn.relu,
                               output_nonlinearity=None,
                               input_shape=(np.prod(env.spec.observation_space.shape),),  # should be correct...
                               batch_normalization=False)
        baseline.initialize_optimizer()

        algo = ActorCritic(env, policy, baseline, delayed_policy=delayed_policy,
                           critic=critic, target_critic=target_critic, soft_target_tau=0.001, n_itr=FLAGS.n_itr,
                           config=config)

        date = calendar.datetime.date.today().strftime('%y-%m-%d')
        if date not in os.listdir(FLAGS.data_dir + '/data'):
            os.mkdir(FLAGS.data_dir + '/data/' + date)
        c = 0
        exp_name = FLAGS.exp_name + '-' + str(c)
        while exp_name in os.listdir(FLAGS.data_dir + '/data/' + date + '/'):
            c += 1
            exp_name = FLAGS.exp_name + '-' + str(c)

        exp_dir = date + '/' + exp_name
        log_dir = osp.join(LOG_DIR, exp_dir)
        policy.set_log_dir(log_dir)

        # run the algorithm and save everything to rllab's pickle format.

        # runner = RLLabRunner(algo, FLAGS, exp_dir)
        # runner.train()
