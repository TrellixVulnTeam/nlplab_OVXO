"""
Create an nlp enviornment
This is a Seq2Seq text enviornment
"""
from __future__ import absolute_import
from __future__ import division

from rllab.envs.base import Env, Step
from rllab.envs.base import EnvSpec
import re
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import numpy as np
import random
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from exps.preprocess import EOS_ID, SOS_ID, PAD_ID, padded, add_sos_eos, refill, tokenize
#from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from exps.preprocess import EOS_ID, SOS_ID
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from sandbox.rocky.tf.spaces.discrete import Discrete
from cached_property import cached_property
from sandbox.rocky.tf.envs.base import TfEnv
import pickle
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.sampler.stateful_pool import ProgBarCounter
import time
from rllab.misc import tensor_utils
import itertools
from rllab.spaces.discrete import Discrete as TheanoDiscrete
from rllab.spaces.box import Box as TheanoBox
from exps.policy import TokenPolicy
import exps.nlc.nlc_data as nlc_data
from exps.nlc.train import create_model

import sys

# set seed
np.random.seed(1234)
random.seed(1234)


# TODO: Today
# TODO: 1. Finish data loading into Env (DONE)
# TODO: 2. Write policy to continue env testing (including having weights imported from Seq2Seq) (ALEX) (DONE)
# TODO: 3. Integrate encoder into Env (DONE), ddist encoding source (DONE)
# TODO:    and have POLICY pull h0 at the right time (would be a batch of h0 though...), examine h0
# TODO: 4. Write Actor-critic algorithm

class DataDistributor(object):
    """
    Pass into NLCEnv, same reference copy
    it is used to store data
    (data is already shuffled during loading..)
    """

    def __init__(self, config):
        self.source = config["source"]
        self.target = config["target"]
        self.source_mask = config["source_mask"]
        self.target_mask = config["target_mask"]
        self.model = config["model"]  # seq2seq model
        self.sess = config["sess"]
        self.target_trackid = -1
        self.rand_inds = np.array(self.shuffle_data())

        # we keep track of how many enviornments are there
        self.n_envs = int(config["batch_size"] / config["max_seq_len"])
        self.n_envs = max(1, min(self.n_envs, 100))

        # this is the current batch
        # policy will use this to pull encoder encodings out of seq2seq model
        self.target_list = range(self.n_envs)

    def encode_source(self):
        # we return all
        # shape: (12, 30) = (n_env, vocab_size?)
        # but remember the source/target shape is (time_step, batch)
        source_tokens = self.source[:, self.get_seq_ids()]  # form a batch
        source_mask = self.source_mask[:, self.get_seq_ids()]

        encoded = self.model.encode(self.sess, source_tokens, source_mask)

        # should know its shape: (30, 12, 100)
        # not really max_len, just time_step determined by preprocessing
        # (max_len, n_env, hidden_states)
        # this returns all encoder's hidden states
        return encoded

    def shuffle_data(self):
        inds = range(self.source.shape[0])
        np.random.shuffle(inds)
        return inds

    def get_seq_ids(self):
        # get all ids from the current "batch" (12 enviornments)
        return self.rand_inds[self.target_list]

    def get_seq_id(self, target_trackid):
        # every env uses their own target_trackid to get from this class
        return self.rand_inds[target_trackid]

    def next_sen(self):
        self.target_trackid += 1
        # if len(self.target_list) != self.n_envs:
        #     self.target_list.append(self.target_trackid)
        # else:
        #     # clean the storage, add the next batch
        #     self.target_list = []
        #     self.target_list.append(self.target_trackid)

        # a better way: when 12, we start new range
        if self.target_trackid % self.n_envs == 0:
            self.target_list = range(self.target_trackid, self.target_trackid + self.n_envs)

        return self.target_trackid


class NLCEnv(Env):
    def __init__(self, distributor, config):
        """
        Train encoder decoder as normal,
        then import embedding into Env
        (but how to use decoder's trained weights to initialize policy?)
        (maybe pass in the tensors to policy....)

        It also interacts with VectorizedSampler

        this enviornment is not a vectorized env (checked by VecotirzedSampler)

        rllab/sampler/utils.py has function rollout()
        calls step()

        algorithm can define a self.max_path_length
        in GAIL it's args.max_traj_len (set this to some number)

        1. Generate a shuffled list of training sets

        2. def step() returns embedding of the previous action (selected vocab)
            requires access to decoder embedding
            - NLCEnv stores decoder embedding now
            - step(action), will have action as a placeholder, and load embedding with it

        3. def step() also returns attention map from the encoder as "info", so that decoder (policy)
            can use it
        4. Override decoder on a CategoricalGaussian policy that has stochastic action
            and deterministic action
                - Form actor network (policy) and critic network (Q) inside the policy
                -
        5. Write train code for encoder-decoder, as well as for policy training

        NOTICE: this env will be copied 12 times, they can't commnicate...

        Parameters
        ----------
        config
            vocab_file: location of vocab file

        """
        self.config = config
        self.distributor = distributor
        self.size = config['gru_size']
        self.vocab_file = config['data_dir'] + "vocab.dat"
        #self.vocab_size = self.count_vocab()
        self.vocab_size = 100
        self.batch_size = config['batch_size']
        self.max_seq_len = config['max_seq_len']
        self.curr_sent = np.zeros(self.max_seq_len, dtype="int32")  # store sentence for one rollout
        self.target_trackid = 0
        self.step_counter = 0
        self.source = config["source"]
        self.target = config["target"]
        self.source_mask = config["source_mask"]
        self.target_mask = config["target_mask"]
        self.model = config["model"]  # seq2seq model
        self.L_dec = config["L_dec"]  # we get numpy form of decoder embedding
        self.L_enc = config["L_enc"]  # same as above
        self.measure = config["measure"]  # "CER" (character error rate) or "BLEU"

        super(NLCEnv, self).__init__()

    def get_target(self):
        # so this way it will be a shuffled example
        return self.target[self.distributor.get_seq_id(self.target_trackid)]

    def step(self, action):
        """
        Parameters
        ----------
        action: [batch_size], integer of the predicted index of tokens

        Returns
        -------
            Should return a batch-size of previous states (generated tokens)
            and the enviornment is terminated when <END> token is generated
            and BLEU reward is calculated and return as well

            (batch_size x max_seq_len)  -> max_seq_len works for both decoder and encoder

            instead of return batch_size
            we return (max_seq_len) action

            sandbox/rocky/tf/samplers/batch_sampler.py
            Can return numpy array, but not TF tensor

            meaning that we must pretrain.

            (observation, reward, done, info)
        """

        done = action == EOS_ID  # policy outputs <EOS>

        self.curr_sent[self.step_counter] += action

        # Traditionally, BLEU is only generated when the entire dev set
        # is generated since BLEU is corpus-level, not really sentence-level
        # compute reward (BLEU score)
        # we can also compute other forms of measure,
        # like how many tokens are correctly predicted

        if done or self.step_counter >= self.max_seq_len:
            # reset() will initialize new_state
            if self.measure == "BLEU":
                # careful out of bounds with self.step_counter + 1
                # we are not passing in weights...meaning it's using a 4-gram model to evaluate this
                # any sentence shorter than 4 words/chars will get score 0
                rew = sentence_bleu([self.get_target()],
                                    self.curr_sent[:self.step_counter + 1].tolist())
            else:
                rew = self.count_cer(self.get_target(), self.curr_sent[:self.step_counter + 1])
            return Step(self.L_dec[action, :], rew, True, init_hidden=None)  # TODO: should I return this observation?
        else:
            # not done, and within limit
            self.step_counter += 1
            return Step(self.L_dec[action, :], 0, False, init_hidden=None)  # action needs to be 0-based

    def count_cer(self, a, b):
        """
        (works with 1-dim array)
        We pad a or b, whoever is shorter
        """
        if a.shape[0] > b.shape[0]:
            b_padded = np.pad(a, (0, a.shape[0] - b.shape[0]), 'constant', constant_values=(-1))
            a_padded = a
        elif a.shape[0] < b.shape[0]:
            b_padded = b
            a_padded = np.pad(a, (0, b.shape[0] - a.shape[0]), 'constant', constant_values=(-1))
        else:
            a_padded = a
            b_padded = b
        return (a_padded == b_padded).sum() / float(a_padded.shape[0])

    def reset(self):
        """
        Returns
        in rllab/sampler/utils, rollout
        rollout is never used in VectorizedSampler
        we are good
        -------
        (start_token, encoder_feature)
        """
        # TODO: will be 12 envs, each is seperate
        self.curr_sent = np.zeros(self.max_seq_len, dtype="int32")
        self.target_trackid = self.distributor.next_sen()
        return self.L_dec[SOS_ID, :]  # start of sentence token

    def count_vocab(self):
        count = 0
        for _ in open(self.vocab_file).xreadlines(): count += 1
        return count

    @cached_property
    def action_space(self):
        # now Policy can call flatten_n() on these classes
        # also it's cached now
        return TheanoDiscrete(self.vocab_size)

    @cached_property
    def observation_space(self):
        """
        Technically, it should be TheanoBox
        since we are sending in word embedding
        """
        return TheanoDiscrete(self.max_seq_len)

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs, max_path_length):
        # this is a weird function....it basically construct itself
        envs = [NLCEnv(self.distributor, self.config) for _ in range(n_envs)]
        return VecEnvExecutor(
            envs=envs,
            max_path_length=max_path_length
        )


def build_data(fnamex, fnamey, num_layers, max_seq_len):
    # We will need to padd because of encoder
    fdx, fdy = open(fnamex), open(fnamey)
    x_token_list = []
    y_token_list = []

    # we need to fill in the entire dataset
    linex, liney = fdx.readline(), fdy.readline()

    while linex and liney:
        x_tokens, y_tokens = tokenize(linex), tokenize(liney)

        # this is not truncating...just ignoring
        if len(x_tokens) < max_seq_len and len(y_tokens) < max_seq_len:
            x_token_list.append(x_tokens)
            y_token_list.append(y_tokens)

        linex, liney = fdx.readline(), fdy.readline()

    y_token_list = add_sos_eos(y_token_list)  # shift y by 1 position
    x_padded, y_padded = padded(x_token_list, num_layers), padded(y_token_list, 1)

    source_tokens = np.array(x_padded).T  # we ARE transposing source!!!! for encoder!!!!
    source_mask = (source_tokens != PAD_ID).astype(np.int32)
    target_tokens = np.array(y_padded)   # but no need to tranpose target...
    target_mask = (target_tokens != PAD_ID).astype(np.int32)

    return source_tokens, source_mask, target_tokens, target_mask

if __name__ == '__main__':
    source_tokens, source_mask, target_tokens, target_mask = build_data(fnamex="/Users/Aimingnie/Documents" +
                             "/School/Stanford/AA228/nlplab/ptb_data/valid.ids.x",
                      fnamey="/Users/Aimingnie/Documents" +
                             "/School/Stanford/AA228/nlplab/ptb_data/valid.ids.x",
                      num_layers=1, max_seq_len=200)

    print source_tokens.shape
    print target_tokens.shape

    # force to use CPU
    s_config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    # with tf.Session(config=s_config) as sess:
    sess = tf.Session(config=s_config)
    print("Creating %d layers of %d units." % (2, 20))
    model = create_model(sess, vocab_size=52, forward_only=False)
    L_dec = model.L_dec.eval(session=sess)
    L_enc = model.L_enc.eval(session=sess)

    config = {
        "gru_size": 20,
        "source": source_tokens,
        "target": target_tokens,
        "source_mask": source_mask,
        "target_mask": target_mask,
        "model": model,
        "L_dec": L_dec,  # vocab_size: 100, hidden_size: 50
        "L_enc": L_enc,
        "measure": "CER",
        'data_dir': "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/",
        "max_seq_len": 10,  # 200
        "batch_size": 128,
        "sess": sess  # tf_session
    }
    ddist = DataDistributor(config)

    # test if ddist can encode batches of source sentences
    encoded = ddist.encode_source()
    print encoded.shape

    # it appears that TFEnv is compatible with NLCEnv...for now
    env = TfEnv(NLCEnv(ddist, config))

    # print env.vectorized  # false
    # print getattr(env, 'vectorized', False)  # false
    # print "wrapped_env: ", getattr(env, 'wrapped_env', False)
    # print env.vec_env_executor  # exist

    print "wrapped wrapped env: ", env.wrapped_env.wrapped_env

    # these code is from vectorized_sampler
    n_envs = int(config["batch_size"] / config["max_seq_len"])
    n_envs = max(1, min(n_envs, 100))

    print n_envs  # 1 / 12 (for max_seq_len 10)

    # it's not parallel, and not that mysterious, since it's always going to be 1
    # so technically it should be alright
    vec_env = env.vec_env_executor(n_envs=n_envs, max_path_length=config["max_seq_len"])

    # we also test if track_y_id is working
    # second vec_env is the VecEnvExecutor

    # vec_env.vec_env.envs[0].reset()
    # vec_env.vec_env.envs[1].reset()
    # vec_env.vec_env.envs[2].reset()

    # print vec_env.vec_env.envs[0].target_trackid
    # print vec_env.vec_env.envs[1].target_trackid
    # print vec_env.vec_env.envs[2].target_trackid

    # we test after vec_env reset(), distributor will have the correct target_list
    vec_env.reset()

    print ddist.target_list

    vec_env.reset()

    print ddist.target_list

    print vec_env.vec_env.envs[1].target_trackid

    # ====== testing vectorized_sampler's def obtain_samples =======

    paths = []
    n_samples = 0
    obses = vec_env.reset()  # TODO: so env's reset() is called before policy, needs to populate it
    dones = np.asarray([True] * vec_env.num_envs)
    running_paths = [None] * vec_env.num_envs

    # pbar = ProgBarCounter(config["batch_size"])
    policy_time = 0
    env_time = 0
    process_time = 0

    print len(obses)
    print obses[0].shape  # 50

    # TODO: now we need policy to continue testing...
    # We first try a toy policy to just make sure NLCEnv works
    policy = TokenPolicy(env.spec)

    # while n_samples < config["batch_size"]:
    #     t = time.time()
    #     policy.reset()
    #     actions, agent_infos = policy.get_actions(obses)
    #
    #     policy_time += time.time() - t
    #     t = time.time()
    #     next_obses, rewards, dones, env_infos = vec_env.step(actions)
    #     env_time += time.time() - t
    #
    #     t = time.time()
    #
    #     agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
    #     env_infos = tensor_utils.split_tensor_dict_list(env_infos)
    #     if env_infos is None:
    #         env_infos = [dict() for _ in range(vec_env.num_envs)]
    #     if agent_infos is None:
    #         agent_infos = [dict() for _ in range(vec_env.num_envs)]
    #     for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
    #                                                                             rewards, env_infos, agent_infos,
    #                                                                             dones):
    #         if running_paths[idx] is None:
    #             running_paths[idx] = dict(
    #                 observations=[],
    #                 actions=[],
    #                 rewards=[],
    #                 env_infos=[],
    #                 agent_infos=[],
    #             )
    #         running_paths[idx]["observations"].append(observation)
    #         running_paths[idx]["actions"].append(action)
    #         running_paths[idx]["rewards"].append(reward)
    #         running_paths[idx]["env_infos"].append(env_info)
    #         running_paths[idx]["agent_infos"].append(agent_info)
    #         if done:
    #             paths.append(dict(
    #                 observations=env.spec.observation_space.flatten_n(running_paths[idx]["observations"]),
    #                 actions=env.spec.action_space.flatten_n(running_paths[idx]["actions"]),
    #                 rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
    #                 env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
    #                 agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
    #             ))
    #             n_samples += len(running_paths[idx]["rewards"])
    #             running_paths[idx] = None
    #     process_time += time.time() - t
    #     # pbar.inc(len(obses))
    #     obses = next_obses
    #
    #     # pbar.stop()

    # ========== Testing sampler/base.py (BaseSampler's) Process_samples() =======

    sess.close()
