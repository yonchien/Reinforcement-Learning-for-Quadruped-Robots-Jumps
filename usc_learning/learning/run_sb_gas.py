"""
Run growing action space with stable baselines ROS interface, no parallel implementation currently


"""

import os
from datetime import datetime
import inspect 
import numpy as np

import gym
import tensorflow as tf
import logging
# get rid of annoying tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# stable baselines
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1,PPO2, SAC
from stable_baselines.common.cmd_util import make_vec_env

from usc_learning.learning.algos.my_ppo2 import MYPPO2


# should really only use quadruped_master, supply configs here or manually
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
#from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config, get_latest_model
from usc_learning.utils.learning_utils import CheckpointCallback
from stable_baselines.bench.monitor import load_results

USE_ROS = False

# 4 has been tested on mac and linux, right numbner of rollouts
NUM_ENVS = 1

LEARNING_ALG = -1
# -1 is MY_PPO2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC

PARALLEL_FLAG = True
LOAD_PREV_MODEL = False
IMITATION_FLAG = True

if USE_ROS:
    from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
    USE_GAS = True
    NUM_ENVS = 1
    PARALLEL_FLAG = False
    LEARNING_ALG = -1


# Set all save locations
# log_dir = "./logs/"
# MODEL_SAVE_FILENAME = log_dir + 'ppo_aliengo'
# ENV_STATS_FILENAME = os.path.join(log_dir, "vec_normalize.pkl")

#monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'
#save_path = './logs/intermediate_models/'+monitor_dir
SAVE_PATH = 'logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(SAVE_PATH, exist_ok=True)
# copy relevant files to save directory to provide snapshot of code changes
copy_files(SAVE_PATH)


def get_model_env(env_config,save_path=None,old_save_path=None,load_model=None,learning_alg=LEARNING_ALG):
    # # copy relevant files to save directory to provide snapshot of code changes
    os.makedirs(save_path, exist_ok=True)
    # copy_files(save_path)
    # checkpoint 
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path,name_prefix='rl_model', verbose=2)

    if USE_ROS:
        env = Monitor(ROSQuadrupedGymEnv(**env_config), filename=os.path.join(save_path,'monitor.csv'), allow_early_resets=True)
    elif IMITATION_FLAG:
        env = Monitor(ImitationGymEnv(**env_config), filename=os.path.join(save_path,'monitor.csv'), allow_early_resets=True)
    else:
        env = Monitor(QuadrupedGymEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
    env = DummyVecEnv( [lambda: env])
    if load_model is not None:
        env = VecNormalize.load(os.path.join(old_save_path,"vec_normalize.pkl"), env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

    # write env configs to reload (i.e if don't need to inspect whole code base)
    write_env_config(save_path,env)

    #def get_model(env,load_model=None,learning_alg=LEARNING_ALG)
    # Custom MLP policy of two layers of size _,_ each with tanh activation function
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200,100])

    #if LEARNING_ALG <= 0:
    # PPO2 (Default)
    # these settings seem to be working well, with NUM_CPU=4
    n_steps = 1024
    nminibatches = int(n_steps / 64)
    learning_rate = lambda f: 1e-4 #5e-4 * f
    ppo_config = {"gamma":0.99, "n_steps":n_steps, "ent_coef":0.0, "learning_rate":learning_rate, "vf_coef":0.5,
                    "max_grad_norm":0.5, "lam":0.95, "nminibatches":nminibatches, "noptepochs":10, "cliprange":0.2, "cliprange_vf":None,
                    "verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":policy_kwargs,
                    "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
    
    # if load_model is not None:
    #     #model = MYPPO2.load(load_model, env,**ppo_config)
    #     model = MYPPO2('MlpPolicy', env, **ppo_config)
    #     # load weights now?>
    #     model.load_parameters(load_model)
    # else: #LEARNING_ALG == -1:
    #     model = MYPPO2('MlpPolicy', env, **ppo_config)

    if load_model is not None:
        #model = MYPPO2.load(load_model, env,**ppo_config)
        model = PPO2('MlpPolicy', env, **ppo_config)
        # load weights now?>
        model.load_parameters(load_model)
    else: #LEARNING_ALG == -1:
        model = PPO2('MlpPolicy', env, **ppo_config)



    return model, env, checkpoint_callback



NUM_ACTION_SPACES = 20
CONTROL_MODE = "IMPEDANCE"
TASK_ENV = "FWD_LOCOMOTION"
GAS_EACH_NUM_TIMESTEPS = 10000 # min number of timesteps per action space trial
#stopper = CustomStopper()
# model_name = None
# monitor_results = None
if IMITATION_FLAG:
    CONTROL_MODE = "TORQUE"
    TASK_MODE = "IL_ADD_Qdes"

best_ep_mean = -np.inf
best_ep_len = 0
old_save_path = None
prev_len = 20
total_time_steps = 0

i = -1


# LOAD PATH
# old_save_path = "logs/intermediate_models/073120030910/12/073120133047"
# i = 12
#i = 3

while i < NUM_ACTION_SPACES:

    # set env config, replace env_config{}
    print('set env config')
    env_config = {"grow_action_space": True,
                    "rl_action_high": (i+1)/NUM_ACTION_SPACES, # will be between [-rl_high,rl_high]
                    "motor_control_mode": CONTROL_MODE,
                    "task_env": TASK_ENV,
                    "task_mode": TASK_MODE,
                    #"USE_ROS": USE_ROS,
                    #"render": True
                    }

    # if i == 0:
    #     model_name = None
    # else:
    #     model_name = get_latest_model(old_save_path)
    try:
        model_name = get_latest_model(old_save_path)
    except:
        model_name = None

    # don't want to overwrite, if staying at same i
    save_path = os.path.join(SAVE_PATH,str(i),datetime.now().strftime("%m%d%y%H%M%S"))

    model, env, checkpoint_callback  = get_model_env(env_config, 
                                                    save_path=save_path, 
                                                    old_save_path=old_save_path,
                                                    load_model=model_name,
                                                    learning_alg=LEARNING_ALG )

    # Learn and save
    model.learn(total_timesteps=GAS_EACH_NUM_TIMESTEPS, log_interval=1,callback=checkpoint_callback)
    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save( os.path.join(save_path, "ppo_aliengo" ) )  #MODEL_SAVE_FILENAME)
    env.save(os.path.join(save_path, "vec_normalize.pkl" )) #  "vec_normalize.pkl"

    # load stats, decide whether mean and len have increased, in order to increase i
    monitor_results = load_results(save_path)
    print('current lens', monitor_results.l.values[-prev_len:])
    print('current avgs', monitor_results.r.values[-prev_len:])
    prev_ep_len_avg = np.mean(monitor_results.l.values[-prev_len:])
    prev_ep_rew_avg = np.mean(monitor_results.r.values[-prev_len:])
    if prev_ep_len_avg >= best_ep_len and (abs(prev_ep_rew_avg - best_ep_mean) < 1 or prev_ep_rew_avg > best_ep_mean):
        best_ep_len = prev_ep_len_avg
        best_ep_mean = prev_ep_rew_avg
        i += 1

    total_time_steps += GAS_EACH_NUM_TIMESTEPS
    print('*'*100)
    print('- '*50)
    print('stats i',i, 'best mean',best_ep_mean, 'best len', best_ep_len)
    print('current', 'mean', prev_ep_rew_avg, 'len', prev_ep_len_avg)
    print('save_path', save_path)
    print('total timesteps', total_time_steps)
    print('- '*50)
    print('*'*100)
    
    old_save_path = save_path#.copy()
