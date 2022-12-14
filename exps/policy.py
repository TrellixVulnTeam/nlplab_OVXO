import numpy as np
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import GRUNetwork, MLP
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.base import StochasticPolicy, Policy
from exps.layers import AttnGRULayer

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides

"""
In order to not change the signature of reset()
when reset() is called
We alter GRUNetwork's h0 (init hidden state)

We pass in env, so that we can use encoder stored in env
to alter reset()
(but now the question is: how does Env keep track of VectorizedSampler?)

"""


# TODO 1: it basically works by now.

# I guess the difficulty is that Policy already assembles layers....

class TokenPolicy(Policy):
    def __init__(self, env_spec):
        super(TokenPolicy, self).__init__(env_spec)

    def reset(self, **kwargs):
        pass

    def get_actions(self, observations):
        flat_obs = np.array(observations, dtype=observations[0].dtype)
        all_input = flat_obs
        actions = np.ones(self.observation_space.n)
        return actions


class CategoricalGRUPolicy(StochasticPolicy, LayersPowered, Serializable):
    """
    This is a customized version of the CategoricalGRUPolicy
    more specifically
    """

    def __init__(
            self,
            name,
            env_spec,
            distributor,
            config,
            hidden_dim=32,
            feature_network=None,
            state_include_action=False,
            hidden_nonlinearity=tf.tanh,
            word_embed_dim=0,
            gru_layer_cls=AttnGRULayer,
    ):
        """
        :param config: has L_dec, L_enc

        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        self.save_name = "policy_gru_categorical"
        self.distributor = distributor
        self.config = config
        n_envs = int(config["batch_size"] / config["max_seq_len"])
        self.n_envs = max(1, min(n_envs, 100))
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Discrete)
            Serializable.quick_init(self, locals())
            super(CategoricalGRUPolicy, self).__init__(env_spec)

            if word_embed_dim != 0:
                obs_dim = word_embed_dim
            else:
                obs_dim = env_spec.observation_space.flat_dim

            action_dim = env_spec.action_space.flat_dim  # over vocab space

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            # old computational graph is meaningless to us now
            # Keep in mind InputLayer doesn't contain any any params, just placeholder

            # l_input = tf.placeholder(shape=(None, None, input_dim))
            # it could be 1, batch-size/n-env, input_dim
            l_input = L.InputLayer(
                shape=(None, None, input_dim),
                name="input"
            )

            # we can add a feature_network
            # but it's just an affine transformation between word_embedding
            # and recurrent layer (we can, but do we need to?)
            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.pack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )

            # TODO: we want to assign tf vars in GRUNetwork (contact Alex)
            prob_network = GRUNetwork(
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=env_spec.action_space.n,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=tf.nn.softmax,
                gru_layer_cls=gru_layer_cls,
                name="prob_network",
                layer_args={"encoder_max_seq_length":config["encoder_max_seq_length"],
                            "n_env":self.n_envs}
            )

            self.prob_network = prob_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            self.f_step_prob = tensor_utils.compile_function(
                [
                    flat_input_var,
                    prob_network.step_prev_state_layer.input_var,
                    # prob_network.recurrent_layer.hs
                ],
                L.get_output([
                    prob_network.step_output_layer,
                    prob_network.step_hidden_layer
                ], {prob_network.step_input_layer: feature_var})
            )

            self.f_output = tensor_utils.compile_function(
                inputs=[
                    self.prob_network.input_layer.input_var,  # (batch_size, time_step, input_dim)
                    # self.prob_network.recurrent_layer.h0_sym, # self.prev_hiddens (used because otherwise there's problem)
                    # prob_network.recurrent_layer.hs,  # self.hs
                ],
                outputs=L.get_output(
                    self.prob_network.output_layer
                )
            )

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            out_layers = [prob_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    def get_action_probs(self, observations):
        """
        This is called by algorithm to obtain an action probability distribution

        observations: (batch_size, time_step, )
        """
        # TODO: let's hope NLC model's weights are shaped the same way as this


    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        obs_var = tf.reshape(obs_var, tf.pack([n_batches, n_steps, -1]))  # since our obs is just (batch, step, dim),
                                                                          # this won't change anything
        obs_var = tf.cast(obs_var, tf.float32)
        if self.state_include_action:
            prev_action_var = tf.cast(state_info_vars["prev_action"], tf.float32)
            all_input_var = tf.concat(2, [obs_var, prev_action_var])
        else:
            all_input_var = obs_var

        # TODO: ho ho ho, this won't work. what is L.get_output() here doing?
        if self.feature_network is None:
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var}
                )
            )
        else:
            flat_input_var = tf.reshape(all_input_var, (-1, self.input_dim))
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var, self.feature_network.input_layer: flat_input_var}
                )
            )

    # this is used to create state_info_vars by algorithms like VPG
    # then pass into dist_info_sym()
    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return []

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        """
        vec_env gets reset first, then policy gets reset
        (only reset hs when it's truly new)
        Also every reset(), loads in h0 and hs

        with correct h0 and hs, policy makes actions
        but what about optimize? When is it called?

        """
        if dones is None:
            dones = [True] * self.n_envs  # for vectorized_enviornment
            # Vectorized_Sampler does pass in dones
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            # it's only called at the VERY FIRST time (of all loops)
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))  # n_env, self.hidden_dim

        # populate self.hs (very inefficient, since we populate it every single time)
        # the below code only populates when DONE is true
        hs = self.distributor.encode_source()

        # assign the weights INTO the network, no need to pass around
        # as a symbolic variable.
        self.prob_network.recurrent_layer.assign_hs_weights(hs)

        self.prev_actions[dones] = 0.
        # hmmmm
        # self.prev_hiddens[dones] = self.prob_network.hid_init_param.eval({
        #     self.prob_network.recurrent_layer.hs: self.hs
        # })  # get_value() maybe direct slicing is faster...

        # we use "dones" because we only load in new hs when it's DONE
        if (dones == True).sum() == dones.shape[0]:
            # print hs.shape (30, 2, 20), which is CORRECT!
            # print "length of done: ", len(dones) = 10, WRONG!
            print "loading in encoder weights of sentences: ", self.distributor.get_seq_ids()
            self.prev_hiddens[dones] = hs[-1,:,:]

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        """

        Parameters
        ----------
        observations: [n_envs, word_dim]
                       each elem is self.observation_space

        return actions: [n_envs, action_dim]
        """
        # flatten_n won't work, it's to make observations one-hot
        # we turn a list of observations into a matrix
        flat_obs = self.observation_space.flatten_n(observations)
        # flat_obs = np.array(observations, dtype="float32")  # no need for this line anymore
        if self.state_include_action:
            raise NotImplementedError  # because action doesn't work
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
        else:
            all_input = flat_obs

        # retrieve self.hs from ddist
        # hs is already loaded into the network through reset()

        probs, hidden_vec = self.f_step_prob(all_input, self.prev_hiddens)
        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        prev_actions = self.prev_actions

        # flatten_n doesn't work :(
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(prob=probs)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)

        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist
