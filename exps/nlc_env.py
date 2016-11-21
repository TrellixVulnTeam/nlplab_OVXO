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
from exps.preprocess import EOS_ID, SOS_ID
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from sandbox.rocky.tf.spaces.discrete import Discrete
from cached_property import cached_property
from sandbox.rocky.tf.envs.base import TfEnv
from exps.normalized_env import normalize
import pickle
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.sampler.stateful_pool import ProgBarCounter
import time
from rllab.misc import tensor_utils
import itertools
from rllab.spaces.discrete import Discrete as TheanoDiscrete

# set seed
np.random.seed(1234)

# TODO: Today
# TODO: 2. Write policy to continue env testing

class DataDistributor(object):
    """
    Pass into NLCEnv, same reference copy
    it is used to store randomly shuffled data
    """
    def __init__(self, config):
        self.source = config["source"]
        self.target = config["target"]
        self.rand_inds = self.shuffle_data()
        self.target_trackid = -1

    def shuffle_data(self):
        inds = range(self.source.shape[0])
        np.random.shuffle(inds)
        return inds

    def get_seq_id(self, target_trackid):
        # every env uses their own target_trackid to get from this class
        return self.rand_inds[target_trackid]

    def next_sen(self):
        self.target_trackid += 1
        return self.target_trackid


class NLCEnv(Env):

    target_trackid = 0  # class variable, will be set by reset() each time

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
        self.vocab_size = self.count_vocab()
        self.batch_size = config['batch_size']
        self.max_seq_len = config['max_seq_len']
        self.curr_sent = np.zeros(self.max_seq_len, dtype="int32")  # store sentence for one rollout
        self.target_trackid = 0
        self.step_counter = 0
        self.source = config["source"]
        self.target = config["target"]
        self.model = config["model"]  # seq2seq model
        self.L_dec = config["L_dec"]  # we get numpy form of decoder embedding
        self.L_enc = config["L_enc"]  # same as above
        self.measure = config["measure"]  # "CER" (character error rate) or "BLEU"

        super(NLCEnv, self).__init__()

    def get_target(self):
        # so this way it will be a shuffled example
        return self.distributor.get_seq_id(self.target_trackid)

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
        info = self.encoder  # TODO: encoder must encode the previous sentence

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
        I think this is where padding comes in
        [batch_size x max_seq_len]

        Returns
        -------
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

    def candy(self):
        return "there is a candy :)"

if __name__ == '__main__':
    config = {
        "gru_size": 100,
        "source": np.ones((20, 5)),
        "target": np.ones((20, 5)),
        "model": None,
        "L_dec": np.ones((100, 50)),  # vocab_size: 100, hidden_size: 50
        "L_enc": None,
        "measure": "CER",
        'data_dir': "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/",
        "max_seq_len": 10,  # 200
        "batch_size": 128
    }
    ddist = DataDistributor(config)
    # it appears that TFEnv is compatible with NLCEnv...for now
    env = TfEnv(normalize(NLCEnv(ddist, config)))

    # print env.vectorized  # false
    # print getattr(env, 'vectorized', False)  # false
    # print "wrapped_env: ", getattr(env, 'wrapped_env', False)
    # print env.vec_env_executor  # exist

    print "wrapped wrapped env: ", env.wrapped_env.wrapped_env

    # these code is from vectorized_sampler
    n_envs = int(config["batch_size"] / config["max_seq_len"])
    n_envs = max(1, min(n_envs, 100))

    print n_envs # 1 / 12 (for max_seq_len 10)

    # it's not parallel, and not that mysterious, since it's always going to be 1
    # so technically it should be alright
    vec_env = env.vec_env_executor(n_envs=n_envs, max_path_length=config["max_seq_len"])

    print vec_env
    # we also test if track_y_id is working
    # second vec_env is the VecEnvExecutor
    vec_env.vec_env.envs[0].reset()
    vec_env.vec_env.envs[1].reset()
    vec_env.vec_env.envs[2].reset()

    print vec_env.vec_env.envs[0].target_trackid
    print vec_env.vec_env.envs[1].target_trackid
    print vec_env.vec_env.envs[2].target_trackid

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

    while n_samples < config["batch_size"]:
        t = time.time()
        policy.reset()
        actions, agent_infos = policy.get_actions(obses)

        policy_time += time.time() - t
        t = time.time()
        next_obses, rewards, dones, env_infos = vec_env.step(actions)
        env_time += time.time() - t

        t = time.time()

        agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
        env_infos = tensor_utils.split_tensor_dict_list(env_infos)
        if env_infos is None:
            env_infos = [dict() for _ in range(vec_env.num_envs)]
        if agent_infos is None:
            agent_infos = [dict() for _ in range(vec_env.num_envs)]
        for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                rewards, env_infos, agent_infos,
                                                                                dones):
            if running_paths[idx] is None:
                running_paths[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                )
            running_paths[idx]["observations"].append(observation)
            running_paths[idx]["actions"].append(action)
            running_paths[idx]["rewards"].append(reward)
            running_paths[idx]["env_infos"].append(env_info)
            running_paths[idx]["agent_infos"].append(agent_info)
            if done:
                paths.append(dict(
                    observations=env.spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                    actions=env.spec.action_space.flatten_n(running_paths[idx]["actions"]),
                    rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                    env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                    agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                ))
                n_samples += len(running_paths[idx]["rewards"])
                running_paths[idx] = None
        process_time += time.time() - t
        # pbar.inc(len(obses))
        obses = next_obses

    # pbar.stop()

    # ========== Testing sampler/base.py (BaseSampler's) Process_samples() =======
