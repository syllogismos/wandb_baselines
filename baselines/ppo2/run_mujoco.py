#!/usr/bin/env python3
import wandb
wandb.init()
import gym, os
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger


def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=3e-4,
                       cliprange=0.2,
                       total_timesteps=num_timesteps)

    return model, env


def main():
    args = mujoco_arg_parser().parse_args()
    wandb.config.update(args)
    wandb.config.algo = 'ppo2'
    logger.configure()
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    env_final = gym.make(args.env)
    video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env_final, base_path=os.path.join(wandb.run.dir, 'humanoid'), enabled=True)

    # obs = env_final.reset()

    if True: # if args.play
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env_final.reset()
        while True:
            actions = model.step(obs)[0]
            o, r, d, i  = env_final.step(actions)
            obs[:] = o
            # env.render()
            video_recorder.capture_frame()
            if d:
                obs[:] = env_final.reset()
                video_recorder.close()
                break
            


if __name__ == '__main__':
    main()
