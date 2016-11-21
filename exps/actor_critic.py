from sandbox.rocky.tf.algos.vpg import VPG
from rllab.misc.overrides import overrides
import numpy as np


class ActorCritic(VPG):
    """
    1. override process samples (use a critic network)
    2. Inside we also embed a target network
    3. Implement a Reward Baseline that doesn't use lgbfs

    small things:
    1. Why use first-order optimizer instead of Adam?

    """
    def __init__(self,
                 env, policy, baseline,
                 target, reward,
                 optimizer=None, optimizer_args=None, **kwargs):
        # TODO: max_path_length: used by vectorized_sampler, must pass in there in construction
        super(ActorCritic, self).__init__(env, policy, baseline,
                                          optimizer, optimizer_args, **kwargs)

        self.reward_model = reward
        self.target_network = target


    def init_opt(self):
        super(ActorCritic, self).init_opt()

    def optimize_policy(self, itr, samples_data):
        """
        This is where

        Parameters
        ----------
        itr
        samples_data

        Returns
        -------

        """
        super(ActorCritic, self).optimize_policy(itr, samples_data)

    @overrides
    def process_samples(self, itr, paths):
        """
        Similar to GAIL implementation,
        we save rewards, but use it to train critic network
        and use critic network's output "reward" to train actor

        Since we ARE using Categorical_gru_policy
        and it is vectorized, this should follow VectorizedSampler

        Returns
        -------

        """
        for path in paths:
            X = np.column_stack((path['observations'], path['actions']))
            # if env returns some ambient reward, we want to ignore these for training.
            # but still keep track of it for diagnostics.
            path['env_rewards'] = path['rewards']

            # compute new rewards with discriminator network. Replace env reward in the rollouts.
            rewards = np.squeeze(self.reward_model.compute_reward(X))
            if rewards.ndim == 0:
                rewards = rewards[np.newaxis]
            path['rewards'] = rewards

        assert all([path['rewards'].ndim == 1 for path in paths])

        return self.sampler.process_samples(itr, paths)