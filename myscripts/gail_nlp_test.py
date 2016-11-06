import gym
import argparse
import calendar

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize

from rllab.envs.gym_env import GymEnv
from rllab.config import LOG_DIR

from sandbox import RLLabRunner

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.gail import GAIL
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy

from sandbox.rocky.tf.core.network import MLP, RewardMLP, BaselineMLP
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

import tensorflow as tf
import numpy as np
import os

import os.path as osp
from rllab import config

parser = argparse.ArgumentParser()
# Logger Params
parser.add_argument('--exp_name',type=str,default='my_exp')
parser.add_argument('--tabular_log_file',type=str,default= 'tab.txt')
parser.add_argument('--text_log_file',type=str,default= 'tex.txt')
parser.add_argument('--params_log_file',type=str,default= 'args.txt')
parser.add_argument('--snapshot_mode',type=str,default='all')
parser.add_argument('--log_tabular_only',type=bool,default=False)
parser.add_argument('--log_dir',type=str)
parser.add_argument('--args_data')

# Environment params
parser.add_argument('--trajdatas',type=int,nargs='+',default=[1,2,3,4,5,6])
parser.add_argument('--n_features',type=int,default=45)
parser.add_argument('--limit_trajs',type=int,default=12000)
parser.add_argument('--max_traj_len',type=int,default=100)  # max length of a trajectory (ts)
parser.add_argument('--env_name',type=str,default="Following")
parser.add_argument('--following_distance',type=int,default=10)
parser.add_argument('--normalize_obs',type=bool,default= True)
parser.add_argument('--normalize_act',type=bool,default=False)
parser.add_argument('--norm_tol',type=float,default=1e-1)
parser.add_argument('--render',type=bool, default= False)

# Env dict
## parameters for the environment

# Model Params
parser.add_argument('--policy_type',type=str,default='mlp')
parser.add_argument('--policy_save_name',type=str,default='policy_gail')
parser.add_argument('--baseline_type',type=str,default='linear')
parser.add_argument('--reward_type',type=str,default='mlp') # adversary / discriminator network
parser.add_argument('--load_policy',type=bool,default=False)

parser.add_argument('--hspec',type=int,nargs='+') # specifies architecture of "feature" networks
parser.add_argument('--p_hspec',type=int,nargs='+',default=[]) # policy layers
parser.add_argument('--b_hspec',type=int,nargs='+',default=[]) # baseline layers
parser.add_argument('--r_hspec',type=int,nargs='+',default=[]) # reward layers

parser.add_argument('--gru_dim',type=int,default=64) # hidden dimension of gru

parser.add_argument('--nonlinearity',type=str,default='tanh')
parser.add_argument('--batch_normalization',type=bool,default=False)

# TRPO Params
parser.add_argument('--trpo_batch_size', type=int, default= 40 * 100)

parser.add_argument('--discount', type=float, default=0.95)
parser.add_argument('--gae_lambda', type=float, default=0.99)
parser.add_argument('--n_iter', type=int, default=500)  # trpo iterations

parser.add_argument('--max_kl', type=float, default=0.01)
parser.add_argument('--vf_max_kl', type=float, default=0.01)
parser.add_argument('--vf_cg_damping', type=float, default=0.01)

parser.add_argument('--trpo_step_size',type=float,default=0.01)

parser.add_argument('--only_trpo',type=bool,default=False)

# GAILS Params
parser.add_argument('--gail_batch_size', type=int, default= 1024)
parser.add_argument('--adam_steps',type=int,default=1)
parser.add_argument('--adam_lr', type=float, default=0.00005)
parser.add_argument('--adam_beta1',type=float,default=0.9)
parser.add_argument('--adam_beta2',type=float,default=0.99)
parser.add_argument('--adam_epsilon',type=float,default=1e-8)
parser.add_argument('--decay_steps',type=int,default=0)
parser.add_argument('--decay_rate',type=float,default=1.0)

parser.add_argument('--hard_freeze',type=bool,default=True)
parser.add_argument('--freeze_upper',type=float,default=1.0)
parser.add_argument('--freeze_lower',type=float,default=0.5)

parser.add_argument('--policy_ent_reg', type=float, default=0.0)
parser.add_argument('--env_r_weight',type=float,default=0.0)

args = parser.parse_args()

assert not args.batch_normalization, "Batch normalization not implemented."

nonlins = {'tanh':tf.nn.tanh,'relu':tf.nn.relu,'elu':tf.nn.elu,'sigmoid':tf.nn.sigmoid}
nonlinearity = nonlins[args.nonlinearity]

if args.hspec is None:
    p_hspec = args.p_hspec
    b_hspec = args.b_hspec
    r_hspec = args.r_hspec
else:
    p_hspec = args.hspec
    b_hspec = args.hspec
    r_hspec = args.hspec

## DEFINE THE ENVIRONMENT
gym.envs.register(
    id="OurSuperCoolEnv-v0",
    entry_point='rllab.envs.nlp_env:OurSuperCoolEnvClass',
    timestep_limit=999,
    reward_threshold=195.0,
)

## TODO : Write function for loading trajectories
expert_data_path = LOG_DIR + '/expert_data.h5'
expert_data, expert_data_stacked = load_trajs(expert_data_path, args.limit_trajs)

initial_obs_mean = expert_data_stacked['exobs_Bstacked_Do'].mean(axis= 0)
initial_obs_std = expert_data_stacked['exobs_Bstacked_Do'].std(axis= 0)
initial_obs_var = np.square(initial_obs_std)

# normalize observations
if args.normalize_obs:
    expert_data = {'obs':(expert_data_stacked['exobs_Bstacked_Do'] - initial_obs_mean) / initial_obs_std}
else:
    expert_data = {'obs':expert_data_stacked['exobs_Bstacked_Do']}

# normalize actions
if args.normalize_act:
    initial_act_mean = expert_data_stacked['exa_Bstacked_Da'].mean(axis= 0)
    initial_act_std = expert_data_stacked['exa_Bstacked_Da'].std(axis= 0)

    expert_data.update({'act':(expert_data_stacked['exa_Bstacked_Da'] - initial_act_mean) / initial_act_std})
else:
    initial_act_mean = 0.0
    initial_act_std = 1.0

    expert_data.update({'act':expert_data_stacked['exa_Bstacked_Da']})

# rllab wrapper for gym env. NOTE: takes obs mean and var to normalize during transitions
g_env = normalize(GymEnv(env_id),
                  initial_obs_mean= initial_obs_mean,
                  initial_obs_var= initial_obs_var,
                  normalize_obs= True,
                  running_obs= False)
env = TfEnv(g_env)

## DEFINE POLICY, ADVERSARY, AND BASELINE NETWORKS
# create policy
if args.policy_type == 'mlp':
    policy = GaussianMLPPolicy('mlp_policy', env.spec, hidden_sizes= p_hspec,
                               std_hidden_nonlinearity=nonlinearity,hidden_nonlinearity=nonlinearity,
                               batch_normalization=args.batch_normalization
                               )

elif args.policy_type == 'gru':
    if p_hspec == []:
        feat_mlp = None
    else: # create feature extracting mlp to feed into gru.
        feat_mlp = MLP('mlp_policy', p_hspec[-1], p_hspec[:-1], nonlinearity, output_activation,
                       input_shape= (np.prod(env.spec.observation_space.shape),),
                       batch_normalization=args.batch_normalization)
    policy = GaussianGRUPolicy(name= 'gru_policy', env_spec= env.spec,
                               hidden_dim= args.gru_dim,
                               feature_network=feat_mlp,
                               state_include_action=False)
else:
    raise NotImplementedError

# create baseline
if args.baseline_type == 'linear':
    baseline = LinearFeatureBaseline(env_spec=env.spec)

elif args.baseline_type == 'mlp':
    baseline = BaselineMLP(name='mlp_baseline',
                           output_dim=1,
                           hidden_sizes= b_hspec,
                           hidden_nonlinearity=nonlinearity,
                           output_nonlinearity=None,
                           input_shape=(np.prod(env.spec.observation_space.shape),),
                           batch_normalization=args.batch_normalization)
    baseline.initialize_optimizer()
else:
    raise NotImplementedError

# create adversary / discriminator network which will act as surrogate reward.
reward = RewardMLP('mlp_reward', 1, r_hspec, nonlinearity,tf.nn.sigmoid,
                   input_shape= (np.prod(env.spec.observation_space.shape) + env.action_dim,),
                   batch_normalization= args.batch_normalization
                   )

## DEFINE TRAINING ALGORITHM: GENERATIVE ADVERSARIAL IMITATION LEARNING
algo = GAIL(
    env=env,
    policy=policy,
    baseline=baseline,
    reward=reward,
    expert_data=expert_data,
    batch_size= args.trpo_batch_size,
    gail_batch_size=args.gail_batch_size,
    max_path_length=args.max_traj_len,
    n_itr=args.n_iter,
    discount=args.discount,
    #step_size=0.01,
    step_size=args.trpo_step_size,
    force_batch_sampler= True,
    whole_paths= True,
    adam_steps= args.adam_steps,
    decay_rate= args.decay_rate,
    decay_steps= args.decay_steps,
    act_mean= initial_act_mean,
    act_std= initial_act_std,
    freeze_upper = args.freeze_upper,
    freeze_lower = args.freeze_lower,
    fo_optimizer_cls= tf.train.AdamOptimizer,
    fo_optimizer_args= dict(learning_rate = args.adam_lr,
                            beta1 = args.adam_beta1,
                            beta2 = args.adam_beta2,
                            epsilon= args.adam_epsilon),
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

# crufty code for saving to h5. We probably don't want this.
date= calendar.datetime.date.today().strftime('%y-%m-%d')
if date not in os.listdir(rllab_path+'/data'):
    os.mkdir(rllab_path+'/data/'+date)
c = 0
exp_name = args.exp_name + '-'+str(c)
while exp_name in os.listdir(rllab_path+'/data/'+date+'/'):
    c += 1
    exp_name = args.exp_name + '-'+str(c)

exp_dir = date+'/'+exp_name
log_dir = osp.join(config.LOG_DIR, exp_dir)
policy.set_log_dir(log_dir)

# run the algorithm and save everything to rllab's pickle format.
runner = RLLabRunner(algo, args, exp_dir)
runner.train()
