import os

# from rllab.algos.cem import CEM
# from rllab.algos.cma_es import CMAES
# from rllab.algos.erwr import ERWR

# os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'

# from rllab.algos.vpg import VPG
# from rllab.algos.tnpg import TNPG
# from rllab.algos.ppo import PPO
# from rllab.algos.trpo import TRPO
# from rllab.algos.reps import REPS
# from rllab.algos.ddpg import DDPG
# from rllab.envs.grid_world_env import GridWorldEnv
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
# from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
# from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
# from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
# from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
# from rllab.exploration_strategies.ou_strategy import OUStrategy
# from rllab.baselines.zero_baseline import ZeroBaseline
# from nose2 import tools
# import numpy as np

# common_batch_algo_args = dict(
#     n_itr=1,
#     batch_size=1000,
#     max_path_length=100,
# )
#
# algo_args = {
#     VPG: common_batch_algo_args,
#     TNPG: dict(common_batch_algo_args,
#                optimizer_args=dict(
#                    cg_iters=1,
#                ),
#                ),
#     TRPO: dict(common_batch_algo_args,
#                optimizer_args=dict(
#                    cg_iters=1,
#                ),
#                ),
#     PPO: dict(common_batch_algo_args,
#               optimizer_args=dict(
#                   max_penalty_itr=1,
#                   max_opt_itr=1
#               ),
#               ),
#     REPS: dict(common_batch_algo_args,
#                max_opt_itr=1,
#                ),
#     DDPG: dict(
#         n_epochs=1,
#         epoch_length=100,
#         batch_size=32,
#         min_pool_size=50,
#         replay_pool_size=1000,
#         eval_samples=100,
#     ),
#     CEM: dict(
#         n_itr=1,
#         max_path_length=100,
#         n_samples=5,
#     ),
#     ERWR: common_batch_algo_args,
# }
#
# polopt_cases = []
# for algo in [VPG, TNPG, PPO, TRPO, CEM, ERWR, REPS]:
#     polopt_cases.extend([
#         (algo, GridWorldEnv, CategoricalMLPPolicy),
#         (algo, CartpoleEnv, GaussianMLPPolicy),
#         (algo, GridWorldEnv, CategoricalGRUPolicy),
#         (algo, CartpoleEnv, GaussianGRUPolicy),
#     ])
#
#
# @tools.params(*polopt_cases)
# def test_polopt_algo(algo_cls, env_cls, policy_cls):
#     print("Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__, policy_cls.__name__))
#     env = env_cls()
#     policy = policy_cls(env_spec=env.spec, )
#     baseline = ZeroBaseline(env_spec=env.spec)
#     algo = algo_cls(env=env, policy=policy, baseline=baseline, **(algo_args.get(algo_cls, dict())))
#     algo.train()
#     assert not np.any(np.isnan(policy.get_param_values()))
#
#
# def test_ddpg():
#     env = CartpoleEnv()
#     policy = DeterministicMLPPolicy(env.spec)
#     qf = ContinuousMLPQFunction(env.spec)
#     es = OUStrategy(env.spec)
#     algo = DDPG(
#         env=env, policy=policy, qf=qf, es=es,
#         n_epochs=1,
#         epoch_length=100,
#         batch_size=32,
#         min_pool_size=50,
#         replay_pool_size=1000,
#         eval_samples=100,
#     )
#     algo.train()


def custom_test_vpg():
    from sandbox.rocky.tf.core.network import GRUNetwork
    import numpy as np
    import tensorflow as tf
    from sandbox.rocky.tf.misc import tensor_utils
    import sandbox.rocky.tf.core.layers as L
    from exps.layers import AttnGRULayer, GRULayer

    # TODO: we are testing whether dist_info_sym() from policy will work
    # since it's tightly related to algorithm, called and used by VPG's init_opt()
    input_dim = 3

    obs_var = tf.placeholder(tf.float32, shape=[None, None, input_dim])

    l_input = L.InputLayer(
        shape=(None, None, input_dim),
        name="input"
    )

    prob_network = GRUNetwork(
        name="prob_network",
        input_shape=(input_dim,),  # feature_dim
        input_layer=l_input,
        output_dim=1,
        hidden_dim=4,
        gru_layer_cls=AttnGRULayer,
        layer_args={"encoder_max_seq_length": 30,
                    "n_env": 1}
    )

    # this should get some error because obs_var is not the only input
    # required to get the output_layer
    probs = L.get_output(
        prob_network.output_layer,
        {l_input: obs_var}
    )

    print probs


if __name__ == '__main__':
    custom_test_vpg()
