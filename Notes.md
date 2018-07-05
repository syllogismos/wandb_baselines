# Introduction

In this blog post I want to give a brief introduction to Reinforcement learning, overview of concepts involved. I cannot obviously cover everything in detail, instead I want to talk about how I learned about it and talk about the resources I found useful to get into it and solve complex environments. I will also discuss briefly various popular RL algorithms.

I played with one simple environment called `MountainCar` and one complex environment called `Humanoid` to test various RL algorithms. Below you can see trained agents of both these environments.

### MountainCar

Gym-id: `MountainCarContinuous-v0`

The goal is to make that wagon reach the top of the mountain. State is the position of the car on the hill at a timestep, reward at each step is x displacement. And action is the force to be applied and in what direction on the car.

![Mountain Car](https://thumbs.gfycat.com/WideeyedUntriedJellyfish-size_restricted.gif)

Here the force availabe to us is not enough to make the car go forward and climb uphill. So the agent has to learn to go back first and use gravity to generate some momentum to reach the top of the hill.

### Humanoid

Gym-id: `RoboschoolHumanoid-v1`

The goal here is to make the humanoid bot walk as far as possible.

![Roboschool Humanoid](https://thumbs.gfycat.com/WaryUnhappyEuropeanfiresalamander-size_restricted.gif)

I will go more into the details of the environments after giving the RL definition.

# Reinfrocement Learning

In laymans terms reinforcement learning is a machine learning problem where you learn based on a reward signal you get based on actions you performed at a given state.

The goal of an RL agent is to figure out what action to take given the current state of the environment to maximize the expected cumulative reward over time.

In the mountain car example above, the goal of the agent is to make the wagon reach the top of the mountain. At every timestep you are given the position of the car, the agent has to figure out how much force and in what direction it has to apply to the car such that the cumulative reward is maximized over time. At every timestep when an action is performend, the environment will give you the next state of the car which is it's position and the reward it got and the next position of the car. This environment is tricky because the force available to us is not enought to make the car go forward and climb uphill. So the agent has to learn to go back first and use gravity to generate some momentum to reach the top of the hill.

![Imgur](https://i.imgur.com/nOx1lE1.png)

Formally RL can be modelled as a Markov Decision Process(MDP) problem.

A Markov Decision Process is a tuple ⟨S, A, P, R, γ⟩
* S is a finite set of states
* A is a finite set of actions
* P is a state transition probability matrix,
* Pa′ =P[St+1=s′|St=s,At=a] ss
* R is a reward function, Ras = E[Rt+1 | St = s,At = a] 
* γ is a discount factor γ ∈ [0, 1].


The main differences and challenges when it comes to RL from other ML problems are
* There is no supervisor, only a single reward signal. When its learning, it has to use this reward signal carefully to explore the environment.
* Feedback is delayed, not instantaneous. Because as you can see from `MountainCar` example it has to get negative rewards initially by going back to get the maximal cumulative reward.
* Time really matters (sequential, non i.i.d data) While exploring the state space, the data you will be collecting is highly correlated, although we make some assumptions, the past observations are not independent of the future observations.
* Agent’s actions affect the subsequent data it receives.


I don't want to go too much into the details but I will list some keywords and describe them briefly that might be helpful when you want to learn and look into more.

# Keywords
## Agent
An RL agent might include one or more of the below components.
* Policy: Agents behaviour function. What action to take given the current state
* Value Function: How good each state is.
* Model: Agents representation of the environment.
## Observations, State
The current state of the environment. In Humanoid environment, the observation might be the position of all the limbs.
## Actions
In case of the pong game, actions might be move left, move right or stay still.
## Reward
It's basically a sigal of how good of an action you took at a given state.
## Discount Factor
It's a proxy to memory of the rewards you get because of the past actions, or how far into the future do you want to consider the rewards.
## Value function
Value function basically describes the expected cumulative reward for being in a given state. It takes a state as an input and tells you how much potential reward is possible. Usually it is a neural network.
## Policy function
Policy function basically gives you what action to take given the state. Its the behaviour of RL agent. This is what we are trying to figure out. 
## Q values
Q values are basically describing at a given state, what actions give you the best expected cumulative reward. One slight but important distinction from value function is that Q = f(s, a)
## Bellman equation
This is what we use to compute values of a state or Q values of a state action pair. Its an iterative algorithm. We use this to solve MDPs.


These are some basic terms you would come across. There is so much research happenning, but the basis of RL is surprisingly easy to understand with simple iterative algorithms and intuitive concepts. You can solve simple discrete environments if you are comfortable with above concepts. It gets harder when you are trying to solve environemnts with observations and actions in continuous space.

# Value of WANDB

Before going into more details about the experiment, I want to discuss how much value I'm getting by using wandb. Just by adding 4 lines(two of them import statements) it made my life so much easier.

Before wandb, everytime I run an experiment, I used to create this terminal in tmux window to follow the progress of the learning for each hyperparameter variant I start in a headless machine.

![Learning progress before wandb](https://i.imgur.com/aCV4rDs.png)

It used to take me more than 20 commands to create this dashboard by grepping the log file carefully, using `grep` `watch` commands and etc. And I have to recreate this terminal dashboard for each variant seperately. And imagine sshing into machine to see how the training is progressing for problems that take more than a day.

Now all it takes is to add a single line of code to track the progress, and I get nice graphs.

And one more thing I get out of the box with wandb is logging of system resources.
On top of the above tmux terminal, I used to have htop in another window to see if my system resources are being used effectively. And the most frustrating thing is when the program crashes silently because of not having enough memory after training for few hours. I used to realize it long after it crashed wasting compute resources and etc.

Usually when you are training hard problems it sometimes takes more than a month testing various algorithms and hyperparameters, so tracking your hyperparameters and its book keeping was a huge pain. 

Now all I have to do is start several variants in multiple machines and I just log in to the wandb dashboard, and see how various algorithms, hyperparameters are performing. I would even get the print logs to the stdout streamed to the dashboard.



## Important Note:
When initializeing wandb through `wandb.init()` do it in the first lines of your script. Just to make sure it comes above all your multiprocessing logic either through imports or in code. Otherwise your program might crash with scary error logs like [this](https://gist.github.com/syllogismos/6a531f06326c6265ac378c290b651421).
```
7fc832ad0000-7fc832ad1000 r--p 00025000 ca:01 27501                      /lib/x86_64-linux-gnu/ld-2.23.so
7fc832ad1000-7fc832ad2000 rw-p 00026000 ca:01 27501                      /lib/x86_64-linux-gnu/ld-2.23.so
7fc832ad2000-7fc832ad3000 rw-p 00000000 00:00 0
7ffef91f7000-7ffef9218000 rw-p 00000000 00:00 0                          [stack]
7ffef939c000-7ffef939f000 r--p 00000000 00:00 0                          [vvar]
7ffef939f000-7ffef93a1000 r-xp 00000000 00:00 0                          [vdso]
ffffffffff600000-ffffffffff601000 r-xp 00000000 00:00 0                  [vsyscall]
Aborted (core dumped)
(rllabpp) ubuntu@ip-172-30-0-78:~/wandb_baselines$ wandb: Program ended.
Exception in thread Thread-3:
Traceback (most recent call last):
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/_pslinux.py", line 1401, in wrapper
    return fun(self, *args, **kwargs)
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/_pslinux.py", line 1583, in create_time
    values = self._parse_stat_file()
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/_common.py", line 338, in wrapper
    return fun(self)
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/_pslinux.py", line 1440, in _parse_stat_file
    with open_binary("%s/%s/stat" % (self._procfs_path, self.pid)) as f:
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/_pslinux.py", line 187, in open_binary
    return open(fname, "rb", **kwargs)
FileNotFoundError: [Errno 2] No such file or directory: '/proc/12624/stat'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/__init__.py", line 363, in _init
    self.create_time()
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/__init__.py", line 694, in create_time
    self._create_time = self._proc.create_time()
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/_pslinux.py", line 1412, in wrapper
    raise NoSuchProcess(self.pid, self._name)
psutil._exceptions.NoSuchProcess: psutil.NoSuchProcess process no longer exists (pid=12624)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/threading.py", line 916, in _bootstrap_inner
    self.run()
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/threading.py", line 864, in run
    self._target(*self._args, **self._kwargs)
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/wandb/stats.py", line 94, in _thread_body
    stats = self.stats()
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/wandb/stats.py", line 153, in stats
    stats["proc.memory.rssMB"] = self.proc.memory_info().rss / 1048576.0
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/wandb/stats.py", line 80, in proc
    return psutil.Process(pid=self.run.pid)
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/__init__.py", line 336, in __init__
    self._init(pid)
  File "/home/ubuntu/anaconda2/envs/rllabpp/lib/python3.6/site-packages/psutil/__init__.py", line 376, in _init
    raise NoSuchProcess(pid, None, msg)
psutil._exceptions.NoSuchProcess: psutil.NoSuchProcess no process found with pid 12624
```

## Feature request
One feature request I would like to have is the ability to enable email or sms notifications on the various metrics, say loss metrics or system metrics. This would be very helpful to notify especially when training takes days. Sometimes I use spot instances, and when they go offline I should be notified based on system metric logs disappearing and the experiment is still not finished. Or memory or cpu go above or below a certain threshold and most importantly your loss metric reaches your sweet spot.

## Small bugs
Most issues I have now are with uploading of files like checkpoints and other files in the experiment directory to the wandb dashboard. For some reason when I checkpoint using tensorflow they are saved in a temporary file and it takes sometime for them to be saved in the actual file. So wandb while traversing the files recognizes the temporary files, but while uploading it fails with an exception and stops uploading the rest of the files and sometimes might not even know the actual checkpoints exist. Maybe right before the uploading logic it should wait for sometime maybe 1-2 seconds for the checkpoints to be saved.

I'm pasting the logs I get when I checkpoint using tensorflow. You can see it recognizes temporary files, but it doesnt event see the actual checkpoint file.

```
wandb:     loss_pol_surr ▁█
wandb:         EpLenMean █▁
wandb:   ev_tdlam_before ▁█
wandb: Waiting for final file modifications.
wandb: Syncing files in wandb/run-20180704_085400-ey2kgq4t:
wandb:   wandb-debug.log
wandb:   config.yaml
wandb:   humanoid_policy.data-00000-of-00001.tempstate14432056802831918837
wandb:   humanoid_policy.index.tempstate5254927434801209082
wandb:   checkpoint.tmp9bc4dd33814742488021e2bbbaab8f00
wandb:   humanoid_policy.meta.tmpb3a5730ced754aadb9f46098d3cdde1d
wandb:   humanoid.mp4
wandb:   humanoid.meta.json
wandb:   wandb-metadata.json
wandb:
wandb: Verifying uploaded files... verified!
```

I also came across EmptyFileException for when file exists with no content. When you are dependent on so many libraries any sort of file can be saved and I just wish wandb is more robust in handling these cases.

But overall I get so much value by just adding these 3 lines below.

```
import wandb; wandb.init()
wandb.config.update() # log hyperparameters
wandb.log() # log training progress
```

# How I learnt RL, Resources.

## [Berkeley's, Artificial Intelligence Course]( https://courses.edx.org/courses/BerkeleyX/CS188x_1/1T2013/20021a0a32d14a31b087db8d4bb582fd/)

This course introduces you to the basics of RL, Markov Decision Processes, Bellman Equation, Policy Iteration algorithms, Value Iteration Algorithms. After this, you can easily solve environments whose observation and action spaces are discrete. It also has a cool assignment where you will play with pac man agent, where you build an agent to solve pacman game.

## [David Silver's Youtube lecture](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)

This course actually goes a little bit deeper into  MDP's, Dynamic programming, Model-Free Prediction, Model-Free Control, Value function approximation, actually introduces Neural Networks as Policy and Value functions, Policy gradient algorithm, Actor-Critic algorithm.

I think this is one of the best RL courses out there, not only because the instructor is behind all the cutting edge research happening in this field, but also because of the questions asked by the students. It might not be as polished as Coursera or any other platforms, the participation by the students makes it so much worth it.

## [An Introduction to Reinforcement Learning, Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
Both the above courses are based on this book. Free book you can download from the above link. Very accessible book. The only place where math gets hard is during the discussion of **Policy Gradient** but it is also the basis for all the cool RL algorithms.

Even with vanilla policy gradient algorithm, it's hard to converge hard continuous environemnts like Humanoid and Atari etc. So instead of implementing them myself I'm using one of the popular RL libraries out there Below I'm describing the experiment setup.

# Experiment Setup.
## Open AI GYM
Basically this library makes lots of environments available that you can play with.
## Baselines, RLLAB
RL libraries from OpenAI, that implements most of the popular RL algorithms, that are easy to hack and play around with. They made sure it's easy to reproduce results from all the RL papers.
## Roboschool
Alternative to MUJOCO environments, I didn't have a license to play with MUJOCO environemtns, so instead I installed Roboschool to play with complex environments like Humanoid and etc.

I'm running these experiments on AWS Ubuntu instances, so also needed to have a xvfb installed, to record how the agent actually performs.

All I have to do is run my python script with `xvfb-run` prefixed like below.
```
xvfb-run -s "-screen 0 1400x900x24" python baselines/ppo1/run_mujoco.py --env RoboschoolHumanoid-v1 --num-timesteps 30000000
```
This attaches a fake monitor that the python script can access and record and capture the environment actually playing.

Because I'm using roboschool, it's slightly different how I capture the frames, unlike with normal gym environments. Usually I just use the `Monitor` wrapper to record the video and progress. For some reason this doesn't work with roboschool environments. Instead I use `VideoRecorder` wrapper to do it manually. Below is a sample code of how I record video of a single episode of an environment.

```
env = gym.make('RoboschoolHumanoid-v1')
total_reward = 0
ob = env.reset()
video = True
if video:
    video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, base_path=os.path.join('/home/ubuntu/wandb_baselines', 'humanoid_run2_%i'%seed), enabled=True)

while True:
    action = pi.act(stochastic=False, ob=ob)[0]
    ob, r, done, _ = env.step(action)
    if video:
        video_recorder.capture_frame()
    tot_r += r
    if done:
        ob = env.reset()  
        if video:
            video_recorder.close()
        print(total_reward)
        break
```
# Description of the environment

# Hyperparameters

# Policy Gradient

# Actor Critic

# DDPG

# TRPO

# PPO1

# PPO2

# Results

