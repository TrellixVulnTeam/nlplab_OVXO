from rllab.misc import ext
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.first_order_optimizer import Solver

from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils

import tensorflow as tf
import numpy as np

class GAIL(TRPO):
    """
    Generative adversarial imitation learning.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            fo_optimizer_cls=None,
            fo_optimizer_args=None,
            reward=None,
            expert_data=None,
            gail_batch_size=100,
            adam_steps=1,
            decay_steps=1,
            decay_rate=1.0,
            act_mean = 0.0,
            act_std = 1.0,
            **kwargs):

        super(GAIL, self).__init__(optimizer=optimizer, optimizer_args=optimizer_args, **kwargs)
        self.reward_model = reward
        self.expert_data = expert_data
        self.gail_batch_size = gail_batch_size

        self.act_mean = act_mean
        self.act_std = act_std

        self.global_step = tf.Variable(0, trainable=False)
        decayed_learning_rate = tf.train.exponential_decay(fo_optimizer_args['learning_rate'], self.global_step,
                                                           decay_steps, decay_rate, staircase=True)
        fo_optimizer_args['learning_rate']= decayed_learning_rate

        self.solver = Solver(self.reward_model, 0.0, 0.0, adam_steps, self.gail_batch_size, None,
                             tf_optimizer_cls= fo_optimizer_cls, tf_optimizer_args= fo_optimizer_args)


    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        ## UPDATE POLICY
        super(GAIL, self).optimize_policy_from_inputs(all_input_values)

        ## UPDATE DISCRIMINATOR
        # "full" policy (pi) and expert (ex) datasets.
        obs_pi = all_input_values[0].reshape((-1, np.prod(self.env.observation_space.shape)))
        act_pi = all_input_values[1].reshape((-1, np.prod(self.env.action_space.shape)))
        
        obs_ex= self.expert_data['obs']
        act_ex= self.expert_data['act']        

        # normalize actions . observations are normalized by environment
        act_pi -= self.act_mean
        act_pi /= self.act_std

        # shuffle and extract batches from pi and ex dataset (according to gail batch size).
        p_ex = np.random.choice(obs_ex.shape[0], size=(self.gail_batch_size,), replace=False)
        p_pi = np.random.choice(obs_pi.shape[0], size=(self.gail_batch_size,), replace=False)

        obs_pi_batch = obs_pi[p_pi]
        act_pi_batch = act_pi[p_pi]

        obs_ex_batch = obs_ex[p_ex]
        act_ex_batch = act_ex[p_ex]

        # stack observations and actions into transition pairs.
        trans_Bpi_Do = np.column_stack((obs_pi_batch,act_pi_batch))
        trans_Bex_Do = np.column_stack((obs_ex_batch,act_ex_batch))
        trans_B_Do = np.row_stack((trans_Bpi_Do,trans_Bex_Do))

        Bpi = trans_Bpi_Do.shape[0] # policy batch size.
        Bex = trans_Bex_Do.shape[0] # expert batch size.

        assert Bpi == Bex # if we want to remove this assumption, we need to weight the labels.

        labels= np.concatenate((np.zeros(Bpi), np.ones(Bex)))[...,None] # 0s for policy, 1s for expert

        # train the discriminator network using supervised backprop / adam.
        self.solver.train(trans_B_Do, labels)
        scores = self.reward_model.compute_score(trans_B_Do)

        self.global_step += 1

        accuracy = ((scores < 0) == (labels == 0)).mean()
        accuracy_for_currpolicy = (scores[:Bpi] <= 0).mean()
        accuracy_for_expert = (scores[Bpi:] > 0).mean()
        assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))

        logger.record_tabular('racc', accuracy)
        logger.record_tabular('raccpi', accuracy_for_currpolicy)
        logger.record_tabular('raccex', accuracy_for_expert)

        return dict()

    @overrides
    def process_samples(self, itr, paths):
        for path in paths:
            X = np.column_stack((path['observations'],path['actions']))
            # if env returns some ambient reward, we want to ignore these for training.
            # but still keep track of it for diagnostics.
            path['env_rewards'] = path['rewards']
            
            # compute new rewards with discriminator network. Replace env reward in the rollouts.
            rewards = np.squeeze( self.reward_model.compute_reward(X) )
            if rewards.ndim == 0:
                rewards = rewards[np.newaxis]
            path['rewards'] = rewards 

        assert all([path['rewards'].ndim == 1 for path in paths])

        return self.sampler.process_samples(itr, paths)

