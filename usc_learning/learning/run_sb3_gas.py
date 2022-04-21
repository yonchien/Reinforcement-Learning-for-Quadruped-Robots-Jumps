"""
Run growing action space with stable baselines ROS interface, no parallel implementation currently

check these: 

120420001859/4/120420001859 (better)
120420001921/4/120420001921


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
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

#from usc_learning.learning.algos.my_ppo2 import MYPPO2


# should really only use quadruped_master, supply configs here or manually
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
#from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config, get_latest_model
#from usc_learning.utils.learning_utils import CheckpointCallback


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model and vec_env parameters every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):# -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):# -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)

            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.training_env.save(stats_path)

            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))

        if self.n_calls % 500 == 0: # self.save_freq < 10000 and 
            # also print out path periodically for off-policy aglorithms: SAC, TD3, etc.
            print('=================================== Save path is {}'.format(self.save_path))
        return True



USE_ROS = False

# 4 has been tested on mac and linux, right numbner of rollouts
NUM_ENVS = 1

LEARNING_ALG = 2
# -1 is MY_PPO2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC
# 3 is TD3

PARALLEL_FLAG = True
LOAD_PREV_MODEL = False
IMITATION_FLAG = False

TEST_RL = True

USE_GPU = True

if USE_GPU:
    gpu_arg = "auto" 
else:
    gpu_arg = "cpu"

if USE_ROS:
    from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
    USE_GAS = True
    NUM_ENVS = 1
    PARALLEL_FLAG = False
    LEARNING_ALG = 2


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
        env = Monitor(QuadrupedGymEnv(**env_config), filename=os.path.join(save_path,'monitor.csv'), allow_early_resets=True)
    env = DummyVecEnv( [lambda: env])
    if load_model is not None:
        env = VecNormalize.load(os.path.join(old_save_path,"vec_normalize.pkl"), env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

    # write env configs to reload (i.e if don't need to inspect whole code base)
    write_env_config(save_path,env,updated_config=env_config)

    #def get_model(env,load_model=None,learning_alg=LEARNING_ALG)
    # Custom MLP policy of two layers of size _,_ each with tanh activation function
    #policy_kwargs = dict(net_arch=[200,100]) # act_fun=tf.nn.tanh, 
    policy_kwargs = dict(net_arch=[512,256]) # act_fun=tf.nn.tanh, 

    #if LEARNING_ALG <= 0:
    # PPO2 (Default)
    # these settings seem to be working well, with NUM_CPU=4
    n_steps = 1024
    nminibatches = int(n_steps / 64)
    learning_rate = lambda f: 1e-4 #5e-4 * f
    ppo_config = {"gamma":0.99, "n_steps":n_steps, "ent_coef":0.0, "learning_rate":learning_rate, "vf_coef":0.5,
                    "max_grad_norm":0.5, "gae_lambda":0.95, 
                     "batch_size":64, #"nminibatches":nminibatches, 
                    "n_epochs":10, "clip_range":0.2, "clip_range_vf":0.2, #None,
                    "verbose":1, "tensorboard_log":None, "_init_setup_model":True, "policy_kwargs":policy_kwargs,
                    "device": gpu_arg}

    # policy_kwargs = dict( net_arch=[200,100]) # activation_fn=tf.nn.tanh,
    policy_kwargs = dict( net_arch=[512,256]) # activation_fn=tf.nn.tanh,
    # sac_config={"gamma":0.99, "learning_rate":3e-4, "buffer_size":100000,
    #          "learning_starts":100, "train_freq":1, "batch_size":64,
    #          "tau":0.005, "ent_coef":'auto', "target_update_interval":1,
    #          "gradient_steps":1, "target_entropy":'auto', "action_noise":None,
    #          #"random_exploration":0.0, 
    #          "verbose":1, "tensorboard_log":None,
    #          "_init_setup_model":True, "policy_kwargs":policy_kwargs, "seed":None, "device": gpu_arg}# "full_tensorboard_log":False,
             # "n_cpu_tf_sess":None}
    # checking tuned from zoo
    sac_config={#"gamma":0.98, 
                "learning_rate":1e-4,#7e-4, 
                "buffer_size":300000,
                "batch_size":256,
                "ent_coef":'auto', 
                "gamma":0.99,#0.98, 
                "tau":0.005,#0.02, 
                "train_freq":1,#64, 
                "gradient_steps":1,#64, 
                "learning_starts": 10000 if not load_model else 0,
                #"use_sde": True,
                # "target_update_interval":1, 
                # "target_entropy":'auto', "action_noise":None,
                "verbose":1, "tensorboard_log":None,
                # "_init_setup_model":True, 
                "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300]), 
                "seed":None, "device": gpu_arg}# "full_tensorboard_log":False,

    n_actions = env.action_space.shape[0]
    noise_std = 0.1
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))

    # td3_config={"gamma":0.99, "learning_rate":3e-4, "buffer_size":100000,
    #         "learning_starts":1000, "train_freq":100, "batch_size":128,
    #         "tau":0.005,
    #         "gradient_steps":100, "action_noise":None,
    #         "random_exploration":0.0, "verbose":1, "tensorboard_log":None,
    #         "_init_setup_model":True, "policy_kwargs":policy_kwargs, "full_tensorboard_log":False,
    #         "seed":None, "n_cpu_tf_sess":None}
    td3_config={#"gamma":0.98, 
                "learning_rate":1e-4, 
                "buffer_size":100000,
                "batch_size":128,
                #"ent_coef":'auto', 
                "gamma":0.99,#0.98, 
                "tau":0.005, 
                "train_freq":-1, 
                "gradient_steps":-1, 
                "learning_starts":1000 if not load_model else 0,
                "action_noise": action_noise,
                # "noise_type": 'normal',
                # "noise_std": 0.1,
                "n_episodes_rollout": 1,
                # "target_update_interval":1, 
                # "target_entropy":'auto', "action_noise":None,
                "verbose":1, "tensorboard_log":None,
                # "_init_setup_model":True, 
                "policy_kwargs": dict(net_arch=[512, 256]), # dict(net_arch=[400, 300]),  
                "seed":None, "device": gpu_arg}# "full_tensorboard_log":False,
    # if load_model is not None:
    #     #model = MYPPO2.load(load_model, env,**ppo_config)
    #     model = MYPPO2('MlpPolicy', env, **ppo_config)
    #     # load weights now?>
    #     model.load_parameters(load_model)
    # else: #LEARNING_ALG == -1:
    #     model = MYPPO2('MlpPolicy', env, **ppo_config)

    if LEARNING_ALG == 0:
        if load_model is not None:
            #model = MYPPO2.load(load_model, env,**ppo_config)
            model = PPO('MlpPolicy', env, **ppo_config)
            # load weights now?>
            model.set_parameters(load_model)
        else: #LEARNING_ALG == -1:
            model = PPO('MlpPolicy', env, **ppo_config)

    elif LEARNING_ALG == 2:
        if load_model is not None:
            #model = MYPPO2.load(load_model, env,**ppo_config)
            model = SAC('MlpPolicy', env, **sac_config)
            # load weights now?>
            model.set_parameters(load_model)
            #model = SAC.load(load_model, env,**sac_config)
            # load replay buffer
            print('*\n'*3, 'LOADING REPLAY BUFFER....','num elements is', model.replay_buffer.pos)
            model.load_replay_buffer(os.path.join(old_save_path,"off_policy_replay_buffer"))
            print('*\n'*3, '....LOADED REPLAY BUFFER', 'num elements is', model.replay_buffer.pos,'\n*3'*3,)
        else: #LEARNING_ALG == -1:
            model = SAC('MlpPolicy', env, **sac_config)

    elif LEARNING_ALG == 3:
        if load_model is not None:
            #model = MYPPO2.load(load_model, env,**ppo_config)
            model = TD3('MlpPolicy', env, **td3_config)
            # load weights now?>
            model.set_parameters(load_model)
            #model = SAC.load(load_model, env,**sac_config)
            # load replay buffer
            model.load_replay_buffer(os.path.join(old_save_path,"off_policy_replay_buffer"))
        else: #LEARNING_ALG == -1:
            model = TD3('MlpPolicy', env, **td3_config)




    return model, env, checkpoint_callback



NUM_ACTION_SPACES = 5
CONTROL_MODE = "IMPEDANCE_POS_ONLY" # "IMPEDANCE_TROT" #
# CONTROL_MODE = "PMTG_JOINT"
# CONTROL_MODE = "IMPEDANCE_POS_ONLY_IK" 
TASK_ENV = "FWD_LOCOMOTION"
# TASK_ENV = "DEFAULT"
# GAS_EACH_NUM_TIMESTEPS = 10000 # min number of timesteps per action space trial
GAS_EACH_NUM_TIMESTEPS = 40000 # [UPDATED] for SAC
#stopper = CustomStopper()
# model_name = None
# monitor_results = None
TASK_MODE = None
if IMITATION_FLAG:
    CONTROL_MODE = "TORQUE"
    TASK_MODE = "IL_ADD_Qdes"

best_ep_mean = -np.inf
best_ep_len = 0
old_save_path = None
prev_len = 20
total_time_steps = 0

# i = -1
i = 0

if TEST_RL:
    i = NUM_ACTION_SPACES - 1 
    GAS_EACH_NUM_TIMESTEPS = 10000000 
    TASK_ENV = "FWD_LOCOMOTION"
    #TASK_ENV = "DEFAULT"

# LOAD PATH
# old_save_path = "logs/intermediate_models/073120030910/12/073120133047"
# i = 12
#i = 3

while i < NUM_ACTION_SPACES:

    # set env config, replace env_config{}
    print('set env config')
    if TEST_RL:
        env_config = {}
    else:
        env_config = {"grow_action_space": True,
                    "rl_action_high": (i+1)/NUM_ACTION_SPACES, # will be between [-rl_high,rl_high]
                    "motor_control_mode": CONTROL_MODE,
                    "task_env": TASK_ENV,
                    "task_mode": TASK_MODE,
                    #"energy_weight": 0.01,
                    # "randomize_dynamics":True,
                    #"USE_ROS": USE_ROS,
                    # "render": True
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
    if LEARNING_ALG > 0:
        # save replay buffer for off policy algorithms
        model.save_replay_buffer(os.path.join(save_path,"off_policy_replay_buffer"))

    # load stats, decide whether mean and len have increased, in order to increase i
    monitor_results = load_results(save_path)
    print('current lens', monitor_results.l.values[-prev_len:])
    print('current avgs', monitor_results.r.values[-prev_len:])
    prev_ep_len_avg = np.mean(monitor_results.l.values[-prev_len:])
    prev_ep_rew_avg = np.mean(monitor_results.r.values[-prev_len:])
    #if prev_ep_len_avg >= best_ep_len and (abs(prev_ep_rew_avg - best_ep_mean) < 1 or prev_ep_rew_avg > best_ep_mean):
    if abs(prev_ep_len_avg - best_ep_len) < 2 and (abs(prev_ep_rew_avg - best_ep_mean) < 2 or prev_ep_rew_avg > best_ep_mean):
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
