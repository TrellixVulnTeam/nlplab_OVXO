"""
Testing OpenAI Gym's copy enviornment
"""
import gym
from sandbox.rocky.tf.envs.base import TfEnv
from exps.normalized_env import normalize
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
import pickle
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

env = gym.make("Copy-v0")
env = TfEnv(env)

# print env.action_space
# print env.observation_space

# turns out Env Spec can't tupled action space
# print env.spec

# policy = CategoricalGRUPolicy(name='gru_policy', env_spec=env.spec,
#                            hidden_dim=args.gru_dim,
#                            feature_network=feat_mlp,
#                            state_include_action=False)

config = {
    "gru_size": 20,
    "measure": "CER",
    'data_dir': "/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/",
    "max_seq_len": 10,  # 200
    "batch_size": 128,
}

n_envs = int(config["batch_size"] / config["max_seq_len"])
n_envs = max(1, min(n_envs, 100))

envs = [env for _ in range(n_envs)]
vec_env = VecEnvExecutor(
    envs=envs,
    max_path_length=config["max_seq_len"]
)

print vec_env.reset()