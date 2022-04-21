import os
from datetime import datetime
import time
import numpy as np
import gym
#
import tensorflow as tf
import logging
# get rid of annoying tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# ray
import ray
from ray import tune
from ray.rllib.agents import ppo, sac
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
# quadruped env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# stable baselines vec env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config

"""
THIS FILE: growing action space (GAS)


Notes:
Make sure to pass the following in config (or significantly increases training time with lower returns)
    "observation_filter": "MeanStdFilter",
    -(this is similar to stable-baseliens vectorized environment, keeps running mean and std of observations)
*OR*
VecNormalize env first, and use that in rllib (this will fix the observation/reward scaling like in SB)
"""
monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'


ALG = "PPO"

# use VecNormalize
USING_VEC_ENV = True
SAVE_RESULTS_PATH = os.path.join('./ray_results',monitor_dir)

# get absolute path to the current directory for saving
script_dir = os.path.dirname(os.path.abspath(__file__))

# for copying all relevant files and VecNormalize params (running mean/std)
save_path = os.path.join(script_dir,SAVE_RESULTS_PATH)
vec_stats = os.path.join(save_path, "vec_normalize.pkl")
os.makedirs(save_path, exist_ok=True)
# copy relevant files to save directory to provide snapshot of code changes
copy_files(save_path)


class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, env_config):
        #env = QuadrupedGymEnv(render=False,hard_reset=True)
        if USING_VEC_ENV:
            # same as in stable baselines
            # print('*\n'*10)
            # print(env_config)
            env = lambda: QuadrupedGymEnv(**env_config)
            env = make_vec_env(env, monitor_dir=env_config['vec_save_path'])

            try: # try to load stats
                env = VecNormalize.load(env_config['vec_load_path'], env)
                env.training = True
            except:
                env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=1, clip_obs=100.)

            self.env = env 
            self._save_vec_stats_counter = 0
            self.vec_stats_file = os.path.join(env_config['vec_save_path'], "vec_normalize.pkl")
            # write env configs to reload (i.e if don't need to inspect whole code base)
            write_env_config(env_config['vec_save_path'],env)
        else:
            self.env = QuadrupedGymEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # print('\n','*'*80)
        # print('obs high',self.observation_space.high)
        # print('obs low', self.observation_space.low)

    def reset(self):
        """reset env """
        obs = self.env.reset()
        if USING_VEC_ENV:
            obs = obs[0]
            self._save_vec_stats_counter += 1
            #save vec normalize stats every so often
            if self._save_vec_stats_counter > 100:
                self.env.save(self.vec_stats_file)
                self._save_vec_stats_counter = 0

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        """step env """
        if USING_VEC_ENV:
            # note the VecEnv list convention
            obs, rew, done, info = self.env.step([action])
            obs = obs[0]
            rew = rew[0]
            done = done[0]
            info = info[0]
            #print(obs, rew, done, info )
        else:
            obs, rew, done, info = self.env.step(action)
        #print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            sys.exit()
        return obs, rew, done, info 


#ray.init(address="auto")
ray.init()


# Neural Network parameters
model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh"} # this is BETTER?? "free_log_std": True,

#################################################################################################
### PPO (currently data stays in buffer for a while?)
#################################################################################################
def env_creator(env_config):
    return DummyQuadrupedEnv(env_config)

#env = env_creator('CustomEnv-v0')
#register_env("DummyQuadrupedEnv",lambda: config, env_creator(config) )
config = ppo.DEFAULT_CONFIG.copy()
NUM_WORKERS = 4
config_PPO={"env": DummyQuadrupedEnv,#"DummyQuadrupedEnv", #DummyQuadrupedEnv,
            "num_gpus": 0,
            "num_workers":NUM_WORKERS,
            "num_envs_per_worker": 1, # to test if same as SB
            "lr": 1e-4,
            "monitor": True,
            "model": model,
            "train_batch_size": NUM_WORKERS*1024, #n_steps,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,#256,# WAS 64#nminibatches, ty 256 after
            "rollout_fragment_length": 1024,#num_samples_each_worker,
            "clip_param": 0.2,
            #"use_kl_loss": False,
            # no real reason to limit this?
            "vf_clip_param": 10,#0.2,#0.2,#1.0, #0.2, # in SB, this is default as the same as policy clip_param
            "vf_loss_coeff": 0.5,
            "lambda":0.95, #lam in SB
            "grad_clip": 0.5, # max_grad_norm in SB
            "kl_coeff": 0.0, # no KL in SB, look into this more
            "kl_target": 0.01,
            "entropy_coeff": 0.0,
            #"observation_filter": "MeanStdFilter", # THIS IS NEEDED IF NOT USING VEC_NORMALIZE
            "clip_actions": True,
            "vf_share_layers": True, # this is true in SB
            "normalize_actions": True, # added
            "preprocessor_pref": "rllib", # what are these even doing? anything?
            "simple_optimizer": False, # test this next
            #"batch_mode": "complete_episodes", #"truncate_episodes", 
            "batch_mode": "truncate_episodes", 
            #"custom_preprocessor": "NoPreprocessor"

            # this seemed to have an effect
            "no_done_at_end": True,
            # "soft_horizon": True,
            # "horizon": None, # changing from 501
            "shuffle_buffer_size": NUM_WORKERS*1024,

            # env config:
            "env_config": {},

        }
config.update(config_PPO)
for k,v in config.items():
    print(k,v)
#sys.exit()


from ray.tune import Stopper
from collections import deque
class CustomStopper(Stopper):
    """Custom stopper, let's stop after some pre-determined number of trials, OR if the mean reward is not moving """
    def __init__(self, iteration_timesteps=1):
        self.should_stop = False
        self.rew_means = deque(maxlen=20)
        self.mean_diff = 1 # check if stagnant - not learning anything
        self._max_timesteps_per_space = iteration_timesteps * 10000
        self._prev_trial_avg = []
        self._total_timesteps = 0

    def __call__(self, trial_id, result):
        """We should only return once the new action space is doing at least as well as the old one. """
        self.should_stop = False
        self.rew_means.extend([result["episode_reward_mean"]])
        self._total_timesteps = result['timesteps_total']
        prev_mean = self.safemean(self.rew_means)
        curr_mean = result["episode_reward_mean"]
        
        #if abs(curr_mean-prev_mean) < self.mean_diff or result['timesteps_total'] > self._max_timesteps_per_space:
        if len(self._prev_trial_avg) != 0:
            # check how we are doing compared to previous trial, and make sure we have done at least number of trials
            if (abs(curr_mean-self._prev_trial_avg[0]) < self.mean_diff or curr_mean > self._prev_trial_avg[0]) \
                and result['timesteps_total'] > self._max_timesteps_per_space:
                self.should_stop = True
                print('******************************\ngrowing action space')
                print('prev_mean', prev_mean, 'curr_mean',curr_mean, 'total timesteps', result['timesteps_total'])
                print('******************************')
        elif result['timesteps_total'] > self._max_timesteps_per_space:
            self.should_stop = True
            print('******************************\ngrowing action space')
            print('prev_mean', prev_mean, 'curr_mean',curr_mean, 'total timesteps', result['timesteps_total'])
            print('******************************')

        print('**'*50,'\n',' -'*50)
        print('current stats to beat:')
        print( 'prev best', self._prev_trial_avg, 'current mean',curr_mean)
        print('last 20:', self.rew_means )
        print(' -'*50,'\n','**'*50)
        # if not self.should_stop and result['foo'] > 10:
        #     self.should_stop = True
        return self.should_stop

    def set_additional_time_steps(self,num_timesteps):
        self.should_stop = False
        self._prev_trial_avg.append(self.safemean(self.rew_means))
        #self._max_timesteps_per_space = num_timesteps
        self._max_timesteps_per_space = self._total_timesteps + num_timesteps

    def safemean(self,xs):
        """Avoid division by zero """
        return np.nan if len(xs) == 0 else np.mean(xs)

    def stop_all(self):
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop

# def stopper(trial_id, result):
#     return result["mean_accuracy"] / result["training_iteration"] > 5
# loop through, increase configuation each time .. 
from usc_learning.utils.file_utils import get_latest_model_rllib, get_latest_directory

def get_checkpoint_from_last_GAS(trial_num):
    """Return latest check point  """
    if trial_num < 0: # first trial
        return None
    # now we have a previous trial
    train_dir = get_latest_directory( os.path.join(SAVE_RESULTS_PATH,str(trial_num),ALG) )
    print('train_dir', train_dir)
    # get checkpoint
    checkpoint = get_latest_model_rllib(train_dir)
    return checkpoint

##################################################################################
# Default and max positions
"""
Should these be percentage based or fixed increases?
If percentage, easy to increase action space accordingly in gym environment
    -but, what about for spaces where the nominal action is not in middle of potential actions? (i.e. PD, impedance, ..)


"""
##################################################################################
#import usc_learning.envs.quadruped_master.configs_a1 as robot_config

########################### POSITIONS
#NOMINAL_JOINT_POS = robot_config.INIT_JOINT_ANGLES
# how much to increase these limits? they are not in center of ranges

########################### TORQUES
#NOMINAL_TORQUES = np.zeros(12)

########################### IMPEDANCE
# NOMINAL_IMPEDANCE_POS = np.array([0., 0., -0.3]*4)
# NOMINAL_IMPEDANCE_FN = np.array([-50]*4) # verify..

# robot_config.NOMINAL_IMPEDANCE_POS = NOMINAL_JOINT_POS
# robot_config.NOMINAL_TORQUES = NOMINAL_TORQUES
# robot_config.NOMINAL_IMPEDANCE_POS = NOMINAL_IMPEDANCE_POS
# robot_config.NOMINAL_IMPEDANCE_FN = NOMINAL_IMPEDANCE_FN

"""
for config, need to pass in robot_config to environment, as well as save/load vec normalize path to Dummy Quadruped
"""
# for loading env configs, vec normalize stats
#stats_path = str(log_dir)
#vec_stats_path = os.path.join(stats_path,"vec_normalize.pkl")
##################################################################################
NUM_ACTION_SPACES = 20
CONTROL_MODE = "IMPEDANCE"
GAS_EACH_NUM_TIMESTEPS = 20000 # min number of timesteps per action space trial
stopper = CustomStopper()

for i in range(NUM_ACTION_SPACES):
    time.sleep(1)
    #stopper = CustomStopper(i+1)
    # set env_config
    
    #tune.run(my_trainable, stop=stopper)

    # set env config, replace env_config{}
    print('set env config')
    config['env_config'] = {"grow_action_space": True,
                            "rl_action_high": (i+1)/NUM_ACTION_SPACES, # will be between [-rl_high,rl_high]
                            #"robot_config": robot_config,
                            "vec_save_path": os.path.join(script_dir,SAVE_RESULTS_PATH,str(i)),
                            "vec_load_path": os.path.join(script_dir,SAVE_RESULTS_PATH,str(i-1),"vec_normalize.pkl"),
                            "motor_control_mode": CONTROL_MODE,
                            "task_env": "STANDUP",
                            #"render": True
                            }

    # get restore path, if available
    #try:


    tune.run(
        "PPO",
        local_dir=os.path.join(SAVE_RESULTS_PATH,str(i)),
        checkpoint_freq=10, 
        checkpoint_at_end=True,
        verbose=2,
        stop=stopper,
        #stop= {"timesteps_total": 10000}, #stopper, # see class
        restore=get_checkpoint_from_last_GAS(i-1),#"~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
        config=config,
    )
    time.sleep(1)
    # stopper.set_max_time_steps((i+2)*GAS_EACH_NUM_TIMESTEPS)
    stopper.set_additional_time_steps(GAS_EACH_NUM_TIMESTEPS)

#run some more at end ? 1 with full?
config['env_config'] = {"grow_action_space": True,
                        "rl_action_high": 5, #i/NUM_ACTION_SPACES, # will be between [-rl_high,rl_high]
                        #"robot_config": robot_config,
                        "vec_save_path": os.path.join(script_dir,SAVE_RESULTS_PATH,str(NUM_ACTION_SPACES)),
                        "vec_load_path": os.path.join(script_dir,SAVE_RESULTS_PATH,str(NUM_ACTION_SPACES-1),"vec_normalize.pkl"),
                        "motor_control_mode": CONTROL_MODE
                        }
# tune.run("PPO",
#             config=config, 
#             local_dir=os.path.join(SAVE_RESULTS_PATH,str(NUM_ACTION_SPACES)), # in root of directory now
#             checkpoint_freq=10, 
#             checkpoint_at_end=True,
#             verbose=2,
#             restore=get_checkpoint_from_last_GAS(NUM_ACTION_SPACES-1),
#             stop= {"timesteps_total": 10000}
#         )
"""
How to organize

have "nominal" action in each range
    -increase over time
    - increase RL from [-0.1, 0.1] to [-1,1] which represent percentage of total 

range = MAX_A - MIN_A = 100%

0.1 * range, needs to be scaled to middle of [MIN_A MAX_A]
policy outputs in [-0.1, 0.1], and we scale according to 0.1 of range around center, and increase from there 

midpoint = MIN_A + (MAX_A - MIN_A)/2
final action = midpoint + 0.5*[ -scale * range, scale * range] 

# Restored previous trial from the given checkpoint
# should stop based on if average reward if plateauing 
tune.run(
    "PPO",
    name="RestoredExp", # The name can be different.
    stop={"timesteps_total": 100000}, # train 5 more iterations than previous
    restore="~/ray_results/Original/PG_<xxx>/checkpoint_5/checkpoint-5",
    config={"env": "CartPole-v0"},
)

"""