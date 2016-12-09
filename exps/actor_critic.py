from sandbox.rocky.tf.algos.vpg import VPG
from rllab.misc.overrides import overrides
import numpy as np
import pickle
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
from rllab.misc import ext


# TODO: 2. add code to pretrain critic

class PreTrainer(object):
    """
    It seems that rllab is bad at allowing pretraining
    we write extra code for this (by borrowing from elsewhere)
    """

    def __init__(self, config, env, policy,
                 distributor, critic):
        super(PreTrainer, self).__init__()
        self.config = config
        self.env = env
        self.policy = policy
        self.distributor = distributor
        self.critic = critic


class ActorCritic(VPG):
    """
    1. override process samples (use a critic network)
    2. Inside we also embed a target network
    3. Implement a Reward Baseline that doesn't use lgbfs

    Many aspects borrowed from DDPG

    """

    def __init__(self,
                 env, policy,
                 # delayed_policy, critic, target_critic,
                 baseline,
                 # soft_target_tau=0.001,
                 optimizer=None, optimizer_args=None, **kwargs):
        # __init__ signature must line up with VPG's
        # otherwise Serialization class won't work...

        super(ActorCritic, self).__init__(env, policy, baseline,
                                          optimizer, optimizer_args, **kwargs)

        self.policy = policy
        self.critic = kwargs["critic"]
        self.soft_target_tau = kwargs["soft_target_tau"]

        # First, create "target" policy and Q functions
        # TODO: 1. this could be broken
        # TODO: since you have to load weights into the policy, why don't you just
        # TODO: load twice? Initialize two policies (also policies...are mutable)
        # TODO: so the pickle would only apply
        # TODO: (create policy and critic seperately)
        self.delayed_policy = kwargs["delayed_policy"]  # might have to use tf.train.Saver()
        self.target_critic = kwargs["target_critic"]
        # TODO: 3. we need to pretrain critic with a fixed actor

        # so that n_envs calculation will be correct
        self.max_path_length = kwargs["config"]['max_seq_len']
        self.batch_size = kwargs["config"]["batch_size"]

    def init_opt(self):
        # ======== VPG's init_opt =======
        is_recurrent = int(self.policy.recurrent)

        # we don't need placeholder from network is because we propogate
        # gradients through a categorical variable

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)  # this uses network to compute probs
        # and sample from probs
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)  # got it from RecurrentCategorical
        # instead of computing it from the GRU output, we are computing loss based on the sampling distribution
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - tf.reduce_sum(logli * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)
        else:
            surr_obj = - tf.reduce_mean(logli * advantage_var)
            mean_kl = tf.reduce_mean(kl)
            max_kl = tf.reduce_max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

        # ======== VPG's init_opt End =======

        # we are updating critic in optimize_policy, not in here
        # y need to be computed first
        # critic_update = self.critic.updates  # qf_loss

    def optimize_policy(self, itr, samples_data):
        """
        This is where we optimize actor and critic
        receives input from sampler/base.py BaseSampler's process_samples()
        when you call optimize_policy,
        policy is already reset() exactly once

        and things should load appropriately. (make sure of this)
        """
        # all_input_values = tuple(ext.extract(
        #     samples_data,
        #     "observations", "actions", "advantages"
        # ))
        # agent_infos = samples_data["agent_infos"]
        # state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        # dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        # all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        # if self.policy.recurrent:
        #     all_input_values += (samples_data["valids"],)
        #
        # rewards = ext.extract(samples_data, "rewards")
        # # q_t =

        # TODO: if we pass in the same ddist, calling reset()
        # TODO: should trigger the same h0/hs. Might need to TEST this!
        self.delayed_policy.reset()

        # ext.extract will put things in a tuple, we don't need the tuple part...
        rewards = ext.extract(samples_data, "rewards")[0]
        observations = ext.extract(samples_data, "observations")[0]  # dimension will work??
        actions = ext.extract(samples_data, "actions")[0]  # dimension will work??

        # so now it should be batched
        # we are going through the batch
        # q: (?, time_steps, )

        # print "reward shape: ", rewards.shape (2, 32)
        # print "policy output shape: ", self.delayed_policy.f_output(observations).shape  (2, 32, 52)
        # print "Q output shape: ",  self.target_critic.compute_reward_sa(observations, actions).shape (2, 32)

        q = rewards + np.sum(self.delayed_policy.f_output(observations) *
                             self.target_critic.compute_reward_sa(observations, actions), axis=2)
        # sum out action_dim

        # then we update critic using the computed q
        self.critic.train(observations, actions, q)

        # then we process the rewards and try to get the advantage
        paths = samples_data["paths"]  # this is the original paths

        for path in paths:
            X = np.column_stack((path['observations'], path['actions']))
            # if env returns some ambient reward, we want to ignore these for training.
            # but still keep track of it for diagnostics.
            path['env_rewards'] = path['rewards']

            # compute_reward returns (max_seq_len,) we squeezed
            # hope this is correct. path['actions'] is one-hot encoding
            rewards = np.sum(self.critic.compute_reward(X) * path['actions'], axis=1)

            # it in critic model
            if rewards.ndim == 0:
                rewards = rewards[np.newaxis]
            path['rewards'] = rewards

        assert all([path['rewards'].ndim == 1 for path in paths])

        # so now the output of critic is baked into rewards
        # have the sampler reprocess the sample!
        samples_data = self.sampler.process_samples(itr, paths)

        # this optimize the policy (actor) (in the end), we update policy
        super(ActorCritic, self).optimize_policy(itr, samples_data)

        # so now normal policy and critic are updated, we update delayed policy, critic
        # you can double check if this is correct
        self.delayed_policy.set_param_values(
            self.delayed_policy.get_param_values() * (1.0 - self.soft_target_tau) +
            self.policy.get_param_values() * self.soft_target_tau)

        self.target_critic.set_param_values(
            self.target_critic.get_param_values() * (1.0 - self.soft_target_tau) +
            self.critic.get_param_values() * self.soft_target_tau)

        # TODO: maybe check if this works!??

    @overrides
    def process_samples(self, itr, paths):
        """
        Now this works.
        """

        # we create the raw, batched paths here

        max_path_length = max([len(path["rewards"]) for path in paths])

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            paths=paths,
        )

        # we still batch together actions and rewards and obs

        return samples_data


if __name__ == '__main__':
    pass
    # we test baseline
