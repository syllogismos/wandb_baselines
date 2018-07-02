#!/usr/bin/env python3

import wandb
wandb.init()
import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()
    model_path = os.path.join(wandb.run.dir, 'humanoid_policy')
    U.save_state(model_path)
    env_final = make_mujoco_env(env_id, seed)
    ob = env_final.reset()
    total_r = 0
    while True:
        action = pi.act(stochastic=False, ob=ob)[0]
        ob, r, done, _ = env_final.step(action)
        total_r += r
        if done:
            ob = env_final.reset()
            break
    print(total_r)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    return pi

def main():
    args = mujoco_arg_parser().parse_args()
    wandb.config.update(args)
    wandb.config.algo = 'ppo1'
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
