

import numpy as np


def test_truncate_paths():
    from rllab.sampler.parallel_sampler import truncate_paths

    paths = [
        dict(
            observations=np.zeros((100, 1)),
            actions=np.zeros((100, 1)),
            rewards=np.zeros(100),
            env_infos=dict(),
            agent_infos=dict(lala=np.zeros(100)),
        ),
        dict(
            observations=np.zeros((50, 1)),
            actions=np.zeros((50, 1)),
            rewards=np.zeros(50),
            env_infos=dict(),
            agent_infos=dict(lala=np.zeros(50)),
        ),
    ]

    truncated = truncate_paths(paths, 130)
    assert len(truncated) == 2
    assert len(truncated[-1]["observations"]) == 30
    assert len(truncated[0]["observations"]) == 100
    # make sure not to change the original one
    assert len(paths) == 2
    assert len(paths[-1]["observations"]) == 50

def test_vectorized_sampler():

    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.envs.normalized_env import normalize
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    import pickle
    import tensorflow as tf
    import time
    from sandbox.rocky.tf.misc import tensor_utils
    import itertools

    config = {
        "max_seq_len": 200,  # This is determined by data preprocessing (for decoder it's 32, cause <start>, <end>)
        "batch_size": 400,  # batch_size must be multiples of max_seq_len (did you mix
        "encoder_max_seq_length": 30
    }

    env = TfEnv(normalize(CartpoleEnv()))

    n_envs = int(config["batch_size"] / config["max_seq_len"])
    n_envs = max(1, min(n_envs, 100))

    print "n_envs: ", n_envs

    envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_envs)]

    vec_env = VecEnvExecutor(
        envs=envs,
        max_path_length=config["max_seq_len"]
    )

    paths = []
    n_samples = 0
    obses = vec_env.reset()
    dones = np.asarray([True] * vec_env.num_envs)
    running_paths = [None] * vec_env.num_envs

    # pbar = ProgBarCounter(config["batch_size"])
    policy_time = 0
    env_time = 0
    process_time = 0

    # We first try a toy policy to just make sure NLCEnv works
    policy = GaussianMLPPolicy(
        name='mlp_policy',
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    # spin up a session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    while n_samples < config["batch_size"]:
        t = time.time()
        policy.reset()  # TODO: this is questionable.
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

    print "num of path: ", len(paths)
    # if we increase batch_size, double it, still works
    # we just have 2 paths instead of 1

    # we stack these two compute reward
    print "path observation shape: ", paths[0]["observations"].shape  # (2, 4)
    print "action observation shape: ", paths[0]["actions"].shape  # (2, 1)


    print "path observation shape: ", paths[1]["observations"].shape  # (4, 4)
    print "action observation shape: ", paths[1]["actions"].shape  # (4, 1)

    # so we should be certain that for each path, it's a rollout

if __name__ == '__main__':
    test_vectorized_sampler()
