import gym, os, wandb
# wandb.init()
from baselines.ppo1.run_mujoco import train
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    pi = train(args.env, num_timesteps=1, seed=args.seed, play=False)
    run_home = '/home/ubuntu/wandb_baselines/wandb/run-20180702_220411-4xtopfue/'
    model_path = run_home + 'humanoid_policy'
    # model_path = '/home/ubuntu/wandb_baselines/wandb/run-20180702_220411-4xtopfue/humanoid_policy'
    U.load_state(model_path)
    env = make_mujoco_env('RoboschoolHumanoid-v1', seed=0)
    tot_r = 0
    ob = env.reset()
    runs = 0
    video = False
    if video:
        video_recorder = gym.wrappers.monitoring.video_recorder(env=env, base_path=os.path.join(run_home, 'humanoid_run_%i'%runs), enabled=True)

    while True:
        action = pi.act(stochastic=False, ob=ob)[0]
        ob, r, done, _ = env.step(action)
        if video:
            video_recorder.capture_frame()
        tot_r += r
        if done:
            
            ob = env.reset()  
            runs += 1
            if video:
                video_recorder.close()
                video_recorder = gym.wrappers.monitoring.video_recorder(env=env, base_path=os.path.join(run_home, 'humanoid_run_%i'%runs), enabled=True)
                
            print(tot_r)
            print("@@@@@@@@@@@@@@@")
        if runs > 3:
            break

if __name__ == '__main__':
    main()