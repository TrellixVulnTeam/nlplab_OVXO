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
from exps.nlc_env import NLCEnv
from exps.preprocess import pair_iter
from exps.nlc.nlc_model import NLCModel
import exps.nlc.nlc_data as nlc_data
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.tf.core.network import MLP, Baseline, GRUNetwork
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.misc.overrides import overrides

"""
Assemble the end-to-end encoder-decoder
train normally
then put the model into policy construction
then train it with
"""

tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")  # 40
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")  # 400
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")  # 3
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/",
                           "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./tmp", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("baseline_type", 'mlp', "linear|mlp")

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
        cost = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


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

    def initialize_optimizer(self):
        self._optimizer = FirstOrderOptimizer('optim')

        optimizer_args = dict(
            loss=self.loss(),
            target=self,
            inputs=[self.input_var, self.target_var],
            network_outputs=[self.output]
        )

        self._optimizer.update_opt(**optimizer_args)

    @overrides
    def predict(self, path):
        # X = np.column_stack((path['observations'], path['actions']))
        X = path['observations']
        return super(BaselineMLP, self).predict(X)

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        # self._regressor.fit(observations, returns.reshape((-1, 1)))
        self._optimizer.optimize([observations, returns[..., None]])


class RewardGRUNetwork(GRUNetwork):
    """
    overrides MLP with methods / properties used in generative adversarial learning.
    """

    def compute_reward(self, X):
        predits = -tf.log(1.0 - self.output)
        # predits = -tf.log(1.0 - tf.sigmoid(self.output))
        Y_p = self._predict(predits, X)
        return Y_p

    def compute_score(self, X):
        """
        predict logits ...
        """
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        # logits = self.output
        Y_p = self._predict(logits, X)
        return Y_p

    def likelihood_loss(self):
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        # logits = L.get_output(self.layers[-1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        # ent_B = tfutil.logit_bernoulli_entropy(logits)
        # self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        return tf.reduce_sum(loss)

    def complexity_loss(self, reg, cmx):
        return tf.constant(0.0)

    def loss(self, reg=0.0, cmx=0.0):
        # logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        # ent_B = tfutil.logit_bernoulli_entropy(logits)
        # self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        # return tf.reduce_sum(loss)
        loss = self.likelihood_loss()
        return loss


if __name__ == '__main__':

    # ======== End2End Normal Training =========

    # Train a translation model using NLC data.
    # Prepare NLC data.
    print("Preparing NLC data in %s" % FLAGS.data_dir)

    x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
        FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size,
        tokenizer=get_tokenizer(FLAGS))
    vocab, _ = nlc_data.initialize_vocabulary(vocab_path)
    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, vocab_size, False)

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
                        (
                            epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, iter_time,
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

        # ======== Actor-critic Training =========
        # note that we are still inside the TF session scope
        # TODO: 1. Build a RewardGRU (critic)
        # TODO: 2. Build a TargetGRU (target/slow critic)
=
        # create a critic network
        reward = RewardMLP('mlp_reward', 1, r_hspec, nonlinearity, tf.nn.sigmoid,
                           input_shape=(np.prod(env.spec.observation_space.shape) + env.action_dim,),
                           batch_normalization=args.batch_normalization)

        # create baseline (paper didn't have a baseline)

        # if FLAGS.baseline_type == 'mlp':
        #     baseline = BaselineMLP(name='mlp_baseline',
        #                            output_dim=1,
        #                            hidden_sizes=b_hspec,
        #                            hidden_nonlinearity=nonlinearity,
        #                            output_nonlinearity=None,
        #                            input_shape=(np.prod(env.spec.observation_space.shape),),
        #                            batch_normalization=args.batch_normalization)
        #     baseline.initialize_optimizer()
        # else:
        #     raise NotImplementedError

        # create reward (critic) (it's only used in algo, nowhere else)


        # create target network

        # create policy (exactly the same as seq2seq mode)
        policy = CategoricalGRUPolicy(name='gru_policy', env_spec=env.spec,
                                      hidden_dim=model.size,
                                      feature_network=None,
                                      state_include_action=False)

