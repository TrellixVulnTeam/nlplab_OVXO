#from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

from sandbox.rocky.tf.algos.trpo import TRPO
#from sandbox.rocky.tf.regressors.deterministic_mlp_regressor import DeterministicMLPRegressor
from sandbox.rocky.tf.envs.base import TfEnv

#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = TfEnv(normalize(CartpoleEnv()))

#policy = GaussianMLPPolicy(
    #env_spec=env.spec,
    ## The neural network policy should have two hidden layers, each with 32 hidden units.
    #hidden_sizes=(32, 32)
#)
policy = GaussianMLPPolicy(
    name = 'mlp_policy',
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()
