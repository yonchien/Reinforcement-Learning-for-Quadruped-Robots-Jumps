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
from ray.rllib.agents import ppo, sac, ddpg
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
# quadruped env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# stable baselines vec env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config
from usc_learning.utils.rllib_utils import DummyQuadrupedEnv, CustomStopper, get_checkpoint_from_last_GAS

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

ALG = "TD3"
USE_ROS = False # test works
USE_IMITATION_ENV = False

# save path
SAVE_RESULTS_PATH = os.path.join('./ray_results',monitor_dir)

# get absolute path to the current directory for saving
script_dir = os.path.dirname(os.path.abspath(__file__))

# for copying all relevant files and VecNormalize params (running mean/std)
save_path = os.path.join(script_dir,SAVE_RESULTS_PATH)
os.makedirs(save_path, exist_ok=True)
# copy relevant files to save directory to provide snapshot of code changes
copy_files(save_path)

#ray.init(address="auto")
ray.init()
init_i = 0

from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_action_dist("SquashedGaussian", SquashedGaussian)
# Neural Network parameters
model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh"} # this is BETTER?? "free_log_std": True,
#model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh", "custom_action_dist": "SquashedGaussian" } # this is BETTER?? "free_log_std": True,

from usc_learning.learning.rllib.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
ModelCatalog.register_custom_model("my_fcnet", MyFullyConnectedNetwork)

model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh", "free_log_std": True,}
model["custom_model"] = "my_fcnet"
model["custom_model_config"] = {"custom_action_std": 1 }

if USE_IMITATION_ENV:
    #model = {"fcnet_hiddens": [512, 256], "fcnet_activation": "tanh", "free_log_std": True} # this is BETTER?? "free_log_std": True,
    model["fcnet_hiddens"] = [512, 256]

#################################################################################################
### PPO (currently data stays in buffer for a while?)
#################################################################################################


if USE_ROS:
    NUM_WORKERS = 1
else:
    NUM_WORKERS = 8

# for testing 
#NUM_WORKERS = 1
# init_i = 5
# SAVE_RESULTS_PATH = os.path.join('./ray_results','080220123506')
# init_i = 9
if NUM_WORKERS == 1:
    num_samples_each_worker = 1024
else:
    num_samples_each_worker = int(4096 / NUM_WORKERS)

if ALG == "PPO":
    config = ppo.DEFAULT_CONFIG.copy()
    config_PPO={"env": DummyQuadrupedEnv,#"DummyQuadrupedEnv", #DummyQuadrupedEnv,
                "num_gpus": 0,
                "num_workers":NUM_WORKERS,
                "num_envs_per_worker": 1, # to test if same as SB
                "lr": 1e-4,
                "monitor": True,
                "model": model,
                "train_batch_size": NUM_WORKERS*num_samples_each_worker, #n_steps,
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 128,#256,# WAS 64#nminibatches, ty 256 after
                "rollout_fragment_length": num_samples_each_worker,#num_samples_each_worker,
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
                #"normalize_actions": True, # added
                "preprocessor_pref": "rllib", # what are these even doing? anything?
                "simple_optimizer": False, # test this next
                #"batch_mode": "complete_episodes", #"truncate_episodes", 
                "batch_mode": "truncate_episodes", 
                #"custom_preprocessor": "NoPreprocessor"

                "log_level": "INFO",

                # this seemed to have an effect
                "no_done_at_end": True,
                # "soft_horizon": True,
                # "horizon": None, # changing from 501
                "shuffle_buffer_size": NUM_WORKERS*num_samples_each_worker,

                # env config:
                "env_config": {},

            }
    config.update(config_PPO)
    # how often to save NN 
    checkpoint_freq=10

elif ALG == "SAC":
    config = sac.DEFAULT_CONFIG.copy()
    config_SAC = { 
        "env": DummyQuadrupedEnv,
        #"horizon": 1000,
        "soft_horizon": False,
        "Q_model": {
          "fcnet_activation": "tanh",
          "fcnet_hiddens": [200,100] if not USE_IMITATION_ENV else [512,256],
          },
        "policy_model": {
          "fcnet_activation": "tanh",
          "fcnet_hiddens": [200,100] if not USE_IMITATION_ENV else [512,256],
          },
        "tau": 0.005,
        # Initial value to use for the entropy weight alpha.
        #"initial_alpha": 1.0,
        "target_entropy": "auto",
        "no_done_at_end": True,
        "n_step": 1,
        "rollout_fragment_length": 1,
        #"prioritized_replay": False,#,True,
        "prioritized_replay": True,
        "train_batch_size": 100,#64,#256,
        "target_network_update_freq": 1,
        # not in original 
        "grad_clip": 0.5,
        "timesteps_per_iteration": 1000,
        "buffer_size": 1000000,
        "learning_starts": 10000,
        "optimization": {
          # "actor_learning_rate": 0.0003,
          # "critic_learning_rate": 0.0003,
          # "entropy_learning_rate": 0.0003,
            "actor_learning_rate": 0.0001,
            "critic_learning_rate": 0.0001,
            "entropy_learning_rate": 0.0001,
          },
        "num_workers": NUM_WORKERS,
        "num_gpus": 0,
        # should not clip actions? - in defaults is False
        #"clip_actions": False,
        #"normalize_actions": True,
        # "evaluation_interval": 1,
        # "metrics_smoothing_episodes": 5,
        }
    config.update(config_SAC)
    checkpoint_freq=100

elif ALG == "TD3":
    config = ddpg.td3.TD3_DEFAULT_CONFIG.copy()
    config_TD3= {
            "env": DummyQuadrupedEnv,
            # largest changes: twin Q functions, delayed policy updates, and target
            # smoothing
            "twin_q": True,
            "policy_delay": 2,
            "smooth_target_policy": True,
            "target_noise": 0.2,#1,#0.2, #0.5,#0.2,
            "target_noise_clip": 0.5,#1, #0.5, #1,#0.5,
            "exploration_config": {
                # TD3 uses simple Gaussian noise on top of deterministic NN-output
                # actions (after a possible pure random phase of n timesteps).
                "type": "GaussianNoise",
                # For how many timesteps should we return completely random
                # actions, before we start adding (scaled) noise?
                "random_timesteps": 5000, #test 10000, #10000,
                # Gaussian stddev of action noise for exploration.
                "stddev": 0.1, #0.1,#0.1,#1, #0.1,
                # Scaling settings by which the Gaussian noise is scaled before
                # being added to the actions. NOTE: The scale timesteps start only
                # after(!) any random steps have been finished.
                # By default, do not anneal over time (fixed 1.0).
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "scale_timesteps": 1e7
            },

            # other changes & things we want to keep fixed:
            # larger actor learning rate, no l2 regularisation, no Huber loss, etc.
            "learning_starts": 5000,#10000,
            "actor_hiddens": [200,100] if not USE_IMITATION_ENV else [512,256],#[400, 300],
            "critic_hiddens":[200,100] if not USE_IMITATION_ENV else [512,256],#[400, 300],
            "actor_hidden_activation": "tanh",
            "critic_hidden_activation": "tanh",
            # Learning rate for the critic (Q-function) optimizer.
            "critic_lr": 1e-4,
            # Learning rate for the actor (policy) optimizer.
            "actor_lr": 1e-4,
            "n_step": 1,
            "gamma": 0.99,
            # "actor_lr": 1e-4, # 1e-3
            # "critic_lr": 1e-4,# 1e-3
            "l2_reg": 0.0,
            "tau": 5e-3,
            "train_batch_size": 100,
            "use_huber": False,
            "target_network_update_freq": 0,
            "num_workers": NUM_WORKERS,
            "num_gpus_per_worker": 0,
            "worker_side_prioritization": False,
            "buffer_size": 1000000,
            "prioritized_replay": False,
            "clip_rewards": False,
            "use_state_preprocessor": False,
            # not in original
            "clip_actions": True,
            "grad_norm_clipping": 0.5,
            }

    config.update(config_TD3)
    checkpoint_freq=100

for k,v in config.items():
    print(k,v)

##################################################################################
NUM_ACTION_SPACES = 20
CONTROL_MODE = "IMPEDANCE_XZ_ONLY"
CONTROL_MODE = "IMPEDANCE"
#CONTROL_MODE = "IMPEDANCE_TROT"
#CONTROL_MODE = "PD"
TASK_ENV = "FWD_LOCOMOTION"
TASK_MODE = None

if USE_IMITATION_ENV:
    CONTROL_MODE = "TORQUE"
    TASK_MODE = "IL_ADD_Qdes"
    #TASK_MODE = "DEFAULT"

# this may be too small, not letting the NN learn anything, and then suddnely having a large action space it does poorly in
GAS_EACH_NUM_TIMESTEPS = 50000 # min number of timesteps per action space trial
stopper = CustomStopper()

for i in range(init_i,NUM_ACTION_SPACES):
    time.sleep(1)
    #stopper = CustomStopper(i+1)
    # set env_config
    print( 'desired vec_save_path:')
    print( os.path.join(script_dir,SAVE_RESULTS_PATH,str(i)) )
    #sys.exit()

    # set env config, replace env_config{}
    print('set env config')
    config['env_config'] = {"grow_action_space": True,
                            "rl_action_high": (i+1)/NUM_ACTION_SPACES, # will be between [-rl_high,rl_high]
                            "scale_gas_action": True,
                            #"robot_config": robot_config,
                            "vec_save_path": os.path.join(script_dir,SAVE_RESULTS_PATH,str(i)),
                            "vec_load_path": os.path.join(script_dir,SAVE_RESULTS_PATH,str(i-1),"vec_normalize.pkl"),
                            "motor_control_mode": CONTROL_MODE,
                            "task_env": TASK_ENV,
                            "USE_ROS": USE_ROS,
                            "USE_IMITATION_ENV": USE_IMITATION_ENV,
                            "task_mode": TASK_MODE,
                            #"render": True
                            }

    # update model std - but after converging may not want to ?
    model["custom_model_config"] = {"custom_action_std": (i+1)/ (NUM_ACTION_SPACES * 2)}
    #model["custom_model_config"] = {"custom_action_std": (i+1)/ (NUM_ACTION_SPACES)}
    #model["custom_model_config"] = {"custom_action_std": 0.2}
    config['model'] = model


    """
    TODO: check what the standard deviation is, and reset to that same one.. but shouldn't that be saved as well? 
    """

    tune.run(
        ALG,
        local_dir=os.path.join(SAVE_RESULTS_PATH,str(i)),
        checkpoint_freq=checkpoint_freq, 
        checkpoint_at_end=True,
        verbose=2,
        stop=stopper,
        #stop= {"timesteps_total": 10000}, #stopper, # see class
        restore=get_checkpoint_from_last_GAS(i-1,save_results_path=SAVE_RESULTS_PATH,alg=ALG),
        config=config,
    )
    time.sleep(1)
    # stopper.set_max_time_steps((i+2)*GAS_EACH_NUM_TIMESTEPS)
    stopper.set_additional_time_steps(GAS_EACH_NUM_TIMESTEPS)

    # need exploration in new action space? 
    # if ALG == "TD3":
    #     config['exploration_config']['random_timesteps'] = 10
    #     config['learning_starts'] = 10
    # if ALG == "SAC":
    #     config['learning_starts'] = 10

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