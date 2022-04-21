import os
from datetime import datetime
import numpy as np
import gym
# ray
import ray
from ray import tune
from ray.rllib.agents import ppo, sac, ddpg, ars, es
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
# quadruped env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
#from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
from usc_learning.envs.car.rcCarFlagRunGymEnv import RcCarFlagRunGymEnv
# stable baselines vec env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config

"""
Notes:
Make sure to pass the following in config (or significantly increases training time with lower returns)
    "observation_filter": "MeanStdFilter",
    -(this is similar to stable-baseliens vectorized environment, keeps running mean and std of observations)
*OR*
VecNormalize env first, and use that in rllib (this will fix the observation/reward scaling like in SB)
"""
monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'


ALG = "PPO"

USE_LSTM = False

USE_IMITATION_ENV = False
USE_CAR_ENV = False
# use VecNormalize
USING_VEC_ENV = False
DONT_NORM_BOOLS = False
SAVE_RESULTS_PATH = os.path.join('./ray_results',monitor_dir)

# get absolute path to the current directory for saving
script_dir = os.path.dirname(os.path.abspath(__file__))

# for saving VecNormalize params (running mean/std)
#monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'
#save_path = 'ray_results/vec_monitor/'+monitor_dir
#save_path = os.path.join(script_dir,save_path)
#vec_stats = os.path.join(save_path, "vec_normalize.pkl")

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
            if USE_IMITATION_ENV:
                env = lambda: ImitationGymEnv()
            else:
                env = lambda: QuadrupedGymEnv(render=False)
            env = make_vec_env(env, monitor_dir=save_path)
            #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)
            env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=1, clip_obs=100.)
            #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)
            self.env = env 
            self._save_vec_stats_counter = 0
            # write env configs to reload (i.e if don't need to inspect whole code base)
            write_env_config(save_path,env)
        else:
            self.env = QuadrupedGymEnv()
            write_env_config(save_path,self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print('\n','*'*80)
        print('obs high',self.observation_space.high)
        print('obs low', self.observation_space.low)

    def reset(self):
        """reset env """
        obs = self.env.reset()
        if USING_VEC_ENV:
            obs = obs[0]
            if DONT_NORM_BOOLS:
                orig_obs = self.env.get_original_obs()
                obs[-4:] = orig_obs[-4:]
            self._save_vec_stats_counter += 1
            #save vec normalize stats every so often
            if self._save_vec_stats_counter > 100:
                self.env.save(vec_stats)
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
            if DONT_NORM_BOOLS:
                orig_obs = self.env.get_original_obs()
                obs[-4:] = orig_obs[-4:]
        else:
            obs, rew, done, info = self.env.step(action)
        #print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            sys.exit()
        return obs, rew, done, info 

# shutdown first, since weird thing between sessions still present?
# there is a memory issue with lots of loading and ctrl^C ending (mem never gets freed)
#ray.shutdown()
#ray.init(address="auto")
ray.init()

from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_action_dist("SquashedGaussian", SquashedGaussian)

from usc_learning.learning.rllib.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
#from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
#import usc_learning.learning.rllib.rllib_helpers.fcnet_me as fcnet_me
ModelCatalog.register_custom_model("my_fcnet", MyFullyConnectedNetwork)
# Neural Network parameters
model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh"} # check free_log_std on/off "free_log_std": True,
#model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh", "custom_action_dist": "SquashedGaussian" } 
# model = {"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh", "free_log_std": True}
# model = {"fcnet_hiddens": [512, 256], "fcnet_activation": "tanh"} # 
# model = {"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh"} #

# model = {"fcnet_hiddens": [200, 100], "fcnet_activation": "tanh", "free_log_std": True,}
# model["custom_model"] = "my_fcnet"
# model["custom_model_config"] = {"custom_action_std": 1 }

if USE_IMITATION_ENV:
    model = {"fcnet_hiddens": [512, 256], "fcnet_activation": "tanh"}

# test to see if we get same thing as in stable baselines
NUM_CPUS = 4
n_steps = 4 * 1024
# should this really be the ENTIRE batch?
nminibatches = 256 #16 #256  #int(n_steps / 64)
num_samples_each_worker = 1024 #512 #1024

if USE_CAR_ENV:
    env = RcCarFlagRunGymEnv
    model = {"fcnet_hiddens": [64, 64], "fcnet_activation": "tanh"}
else:
    env = DummyQuadrupedEnv

if USE_LSTM:
    model = {"use_lstm": True}

#################################################################################################
### SAC
#################################################################################################
if ALG == "SAC":
    config = sac.DEFAULT_CONFIG.copy()
    # config_SAC= {
    #         "env": DummyQuadrupedEnv,
    #         # === Model ===
    #         "twin_q": True,
    #         "use_state_preprocessor": False,
    #         "Q_model": {
    #             #"fcnet_activation": "tanh",
    #             "fcnet_hiddens": [200, 100],
    #             },
    #         "policy_model": {
    #             #"fcnet_activation": "tanh",
    #             "fcnet_hiddens": [200, 100],
    #             },

    #         "optimization": {
    #             "actor_learning_rate": 3e-4,
    #             "critic_learning_rate": 3e-4,
    #             "entropy_learning_rate": 3e-4,
    #             },
    #          #"rollout_fragment_length": 200,
    #          #"train_batch_size": 1024,

    #          #"buffer_size": 10000,
    #          "buffer_size": 1000000,
    #          #"soft_horizon": False,

    #          #"grad_clip": 1,
    #          "tau": 0.005,
    #          "target_entropy": "auto",
    #          "no_done_at_end": True,
    #          "n_step": 1,
    #          "prioritized_replay": True,
    #          "normalize_actions": True,
    #          #hori
    #          "rollout_fragment_length": 1,
    #          "train_batch_size": 256, # 64
    #          "target_network_update_freq": 1,
    #          "timesteps_per_iteration": 1000, #501
    #          "learning_starts": 10000,
    #          "num_workers": 0,
    #          "num_gpus": 0,
    #          # "evaluation_interval": 1,
    #          # "metrics_smoothing_episodes": 5,
    #         }
    
    # test
    # config_SAC = { 
    #     "env": DummyQuadrupedEnv,
    #     #"horizon": 1000,
    #     "soft_horizon": False,
    #     "Q_model": {
    #       "fcnet_activation": "relu",
    #       "fcnet_hiddens": [200, 100],
    #       },
    #     "policy_model": {
    #       "fcnet_activation": "relu",
    #       "fcnet_hiddens": [200, 100],
    #       },
    #     "tau": 0.005,
    #     "target_entropy": "auto",
    #     "no_done_at_end": True,
    #     "n_step": 1,
    #     "rollout_fragment_length": 1,
    #     "prioritized_replay": True,
    #     "train_batch_size": 200,#256,
    #     "target_network_update_freq": 1,
    #     "timesteps_per_iteration": 1000,
    #     "learning_starts": 10000,
    #     "optimization": {
    #       # "actor_learning_rate": 0.0003,
    #       # "critic_learning_rate": 0.0003,
    #       # "entropy_learning_rate": 0.0003,
    #         "actor_learning_rate": 0.001,
    #         "critic_learning_rate": 0.001,
    #         "entropy_learning_rate": 0.001,
    #       },
    #     "num_workers": 0,
    #     "num_gpus": 0,
    #     "clip_actions": False,
    #     "normalize_actions": True,
    #     "no_done_at_end": True,
    #     "evaluation_interval": 1,
    #     "metrics_smoothing_episodes": 5,
    #     }

    # test 2 - like in SB
    config_SAC = { 
        "env": DummyQuadrupedEnv,
        #"horizon": 1000,
        "soft_horizon": False,
        "Q_model": {
          "fcnet_activation": "tanh",
          "fcnet_hiddens": [200, 100],
          },
        "policy_model": {
          "fcnet_activation": "tanh",
          "fcnet_hiddens": [200, 100],
          },
        "tau": 0.005,
        # Initial value to use for the entropy weight alpha.
        #"initial_alpha": 1.0,
        "target_entropy": "auto",
        "no_done_at_end": True,
        "n_step": 1,
        "rollout_fragment_length": 1,
        "prioritized_replay": False,#,True,
        "train_batch_size": 100,#64,#256,
        "target_network_update_freq": 1,
        # not in original 
        "grad_clip": 0.5,
        "timesteps_per_iteration": 1000,
        "buffer_size": 100000,
        "learning_starts": 10000,
        "optimization": {
          # "actor_learning_rate": 0.0003,
          # "critic_learning_rate": 0.0003,
          # "entropy_learning_rate": 0.0003,
            "actor_learning_rate": 0.0001,
            "critic_learning_rate": 0.0001,
            "entropy_learning_rate": 0.0001,
          },
        "num_workers": 0,
        "num_gpus": 0,
        # should not clip actions? - in defaults is False
        #"clip_actions": False,
        #"normalize_actions": True,
        # "evaluation_interval": 1,
        # "metrics_smoothing_episodes": 5,
        }
    config.update(config_SAC)
    # import pdb
    # pdb.set_trace()
    tune.run("SAC",config=config,#config_SAC, 
                    local_dir=SAVE_RESULTS_PATH, 
                    checkpoint_freq=100, 
                    verbose=2,
                    stop= {"timesteps_total": 10000000}
                    )
    ray.shutdown()
    sys.exit()

#################################################################################################
### TD3
#################################################################################################
if ALG == "TD3":
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
                "random_timesteps": 10000, #10000,
                # Gaussian stddev of action noise for exploration.
                "stddev": 0.1, #0.1,#0.1,#1, #0.1,
                # Scaling settings by which the Gaussian noise is scaled before
                # being added to the actions. NOTE: The scale timesteps start only
                # after(!) any random steps have been finished.
                # By default, do not anneal over time (fixed 1.0).
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "scale_timesteps": 1,#1e7
            },

            # other changes & things we want to keep fixed:
            # larger actor learning rate, no l2 regularisation, no Huber loss, etc.
            "learning_starts": 10000,#10000,
            "actor_hiddens": [200,100],#[400, 300],
            "critic_hiddens": [200,100],#[400, 300],
            "actor_hidden_activation": "tanh",
            "critic_hidden_activation": "tanh",
            "n_step": 1,
            "gamma": 0.99,
            "actor_lr": 1e-4, # 1e-3
            "critic_lr": 1e-4,# 1e-3
            "l2_reg": 0.0,
            "tau": 5e-3,
            "train_batch_size": 100,
            "use_huber": False,
            "target_network_update_freq": 0,
            "num_workers": 8,
            "num_gpus_per_worker": 0,
            "worker_side_prioritization": False,
            "buffer_size": 1000000,
            "prioritized_replay": False,
            "clip_rewards": False,
            "use_state_preprocessor": False,
            "clip_actions": True,
            }

    config.update(config_TD3)
    tune.run("TD3",config=config,#config_SAC, 
                    local_dir=SAVE_RESULTS_PATH, 
                    checkpoint_freq=100, 
                    verbose=2,
                    stop= {"timesteps_total": 10000000}
                    )
    ray.shutdown()
    sys.exit()

#################################################################################################
### ARS
#################################################################################################
NUM_CPUS = 8
num_samples_each_worker = int(4096 / NUM_CPUS)
if ALG == "ARS":
    config = ars.DEFAULT_CONFIG.copy()
    config_ARS= {
            "env": DummyQuadrupedEnv,
            "monitor": True,
            #"num_workers": 2,
            "num_gpus": 0,

            # # defaults here?
            # "action_noise_std": 0.0,
            # "noise_stdev": 0.02,  # std deviation of parameter noise
            # "num_rollouts": 1, #32,  # number of perturbs to try
            # "rollouts_used": 1, #32,  # number of perturbs to keep in gradient estimate
            # "num_workers": 2,
            # "sgd_stepsize": 0.01,  # sgd step-size
            # #"observation_filter": "MeanStdFilter",
            # "noise_size": 250000000,
            # "eval_prob": 0.03,  # probability of evaluating the parameter rewards
            # "report_length": 10,  # how many of the last rewards we average over
            # "offset": 0,
            # # ARS will use Trainer's evaluation WorkerSet (if evaluation_interval > 0).
            # # Therefore, we must be careful not to use more than 1 env per eval worker
            # # (would break ARSPolicy's compute_action method) and to not do obs-
            # # filtering.
            # "evaluation_config": {
            #     "num_envs_per_worker": 1,
            #     "observation_filter": "NoFilter"
            # },

            # Use the new "trajectory view API" to collect samples and produce
            # model- and policy inputs.
            # TODO: update rllib
            #"_use_trajectory_view_api": True,


            "action_noise_std": 0.01,
            "noise_stdev": 0.02,  # std deviation of parameter noise
            "num_rollouts": 100, #32,  # number of perturbs to try
            "rollouts_used": 100, #32,  # number of perturbs to keep in gradient estimate
            "num_workers": NUM_CPUS,
            "sgd_stepsize": 0.01,  # sgd step-size
            #"observation_filter": "MeanStdFilter",
            "noise_size": 250000000,
            "eval_prob": 0.03,  # probability of evaluating the parameter rewards
            "report_length": 10,  # how many of the last rewards we average over
            "offset": 0,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "observation_filter": "NoFilter"
            },

            # added
            "train_batch_size": NUM_CPUS*num_samples_each_worker, #4096, #n_steps,
            "rollout_fragment_length": num_samples_each_worker,#102
            "normalize_actions": True,
            "shuffle_buffer_size": NUM_CPUS*num_samples_each_worker, #4096,
            }

    config.update(config_ARS)
    # tune.run("ARS",config=config, 
    #                 local_dir=SAVE_RESULTS_PATH, 
    #                 checkpoint_freq=10, 
    #                 verbose=2,
    #                 stop= {"timesteps_total": 10000000}
    #                 )

    from ray.tune.logger import pretty_print
    trainer = ars.ARSTrainer(config=config, env=DummyQuadrupedEnv)
    # trainer = ars.ARSTrainer(config=config, 
    #                         local_dir=SAVE_RESULTS_PATH,
    #                         verbose=2,
    #                         env=DummyQuadrupedEnv)
    for i in range(500):
        result = trainer.train()
        print(pretty_print(result))
        if i % 10 == 0:
            checkpoint = trainer.save(checkpoint_dir=SAVE_RESULTS_PATH)
            print("checkpoint saved at", checkpoint)
    ray.shutdown()
    sys.exit()

#################################################################################################
### ES
#################################################################################################
NUM_CPUS = 8
num_samples_each_worker = int(4096 / NUM_CPUS)

if ALG == "ES":
    config = es.DEFAULT_CONFIG.copy()
    config_ES= {
            "monitor": True,
            "env": DummyQuadrupedEnv,
            # #"num_workers": 2,
            # "num_gpus": 0,

            # # defaults here?
            # "action_noise_std": 0.01,
            # "l2_coeff": 0.005,
            # "noise_stdev": 0.02,
            # "episodes_per_batch": 1000,
            # "train_batch_size": 10000,
            # "eval_prob": 0.003,
            # "return_proc_mode": "centered_rank",
            # "num_workers": 10,
            # "stepsize": 0.01,
            # "observation_filter": "MeanStdFilter",
            # "noise_size": 250000000,
            # "report_length": 10,
            #         #"_use_trajectory_view_api": True,

            "num_gpus": 0,
            "model": model,
            "action_noise_std": 0.01,
            "l2_coeff": 0.05,
            "noise_stdev": 0.02,
            "episodes_per_batch": 100,
            "eval_prob": 0.003,
            "return_proc_mode": "centered_rank",
            "num_workers": NUM_CPUS,
            "observation_filter": "MeanStdFilter",
            "noise_size": 250000000,
            "report_length": 10,
            "stepsize": 0.03,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "observation_filter": "NoFilter"
            },

            # added
            "train_batch_size": NUM_CPUS*num_samples_each_worker, #4096, #n_steps,
            "rollout_fragment_length": num_samples_each_worker,#102
            "normalize_actions": True,
            "shuffle_buffer_size": NUM_CPUS*num_samples_each_worker, #4096,
            }

    config.update(config_ES)
    tune.run("ES",config=config, 
                    local_dir=SAVE_RESULTS_PATH, 
                    checkpoint_freq=10, 
                    verbose=2,
                    stop= {"timesteps_total": 10000000}
                    )
    ray.shutdown()
    sys.exit()
#################################################################################################
### MADDPG
#################################################################################################
"""
config_MADDPG = {
        "env": DummyQuadrupedEnv,
         "actor_hiddens": [200, 100],
         "actor_hidden_activation": "tanh",
         "critic_hiddens": [200,100],
         "critic_hidden_activation": "tanh",
         "actor_lr": 1e-4,
         "critic_lr": 1e-4,
         "rollout_fragment_length": 1024,
         "train_batch_size": 4096,
         "num_workers": 4
        }
tune.run("contrib/MADDPG",config=config_MADDPG, 
                local_dir='./ray_results/', 
                checkpoint_freq=10, 
                verbose=2,
                #stop= {"stop_timesteps": 1000000}
                )
"""
#################################################################################################
### DDPPO (works)
#################################################################################################
"""
config = ppo.ddppo.DEFAULT_CONFIG.copy()
config_DDPPO = {
    "env": DummyQuadrupedEnv,
    "num_workers":4,
    "rollout_fragment_length": 1024, #num_samples_each_worker,
    "num_envs_per_worker": 1,
    "sgd_minibatch_size": 64, # was 256
    "num_gpus_per_worker": 0,
    "num_sgd_iter": 10,
    "lr": 1e-4,
    "model": model,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "vf_clip_param": 0.2,
    "lambda":0.95, #lam in SB
    "grad_clip": 0.5, # max_grad_norm in SB
    "kl_coeff": 0.0,
    "observation_filter": "MeanStdFilter",
}
config.update(config_DDPPO)
tune.run("DDPPO",config=config, 
                local_dir='./ray_results/', 
                checkpoint_freq=10, 
                verbose=2,
                )
"""
#################################################################################################
### APPO (not working)
#################################################################################################
"""
config = ppo.appo.DEFAULT_CONFIG.copy()
config_APPO = {
        "env": DummyQuadrupedEnv,
        "num_gpus": 0,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "vtrace": True,
        "lr": 1e-4,
        "monitor": True,
        "model": model,
        "train_batch_size": 4096, #n_steps,
        "num_sgd_iter": 10,
        "minibatch_buffer_size": 64,
        #"learner_queue_size": nminibatches,
        #"sgd_minibatch_size": nminibatches, # how to set?
        "rollout_fragment_length": 1024, #num_samples_each_worker,
        "clip_param": 0.2,
        #"vf_clip_param": 0.2,
        "vf_loss_coeff": 0.5,
        "gamma": 0.99,
        "lambda":0.95, #lam in SB
        "grad_clip": 0.5, # max_grad_norm in SB
        "kl_coeff": 0.0, # no KL in SB, look into this more
        "use_kl_loss": False,
        "opt_type": "adam",
        "entropy_coeff": 0.0,
        "observation_filter": "MeanStdFilter",
        #IMPALA related
        "broadcast_interval": 1,
        "max_sample_requests_in_flight_per_worker": 1,
        "num_data_loader_buffers": 1,
        "batch_mode": "truncate_episodes"
        }
config.update(config_APPO)
tune.run("APPO",config=config, 
                local_dir='./ray_results/', 
                checkpoint_freq=10, 
                verbose=2)
"""
#################################################################################################
### PPO (currently data stays in buffer for a while?)
#################################################################################################
config = ppo.DEFAULT_CONFIG.copy()

# testenv = DummyQuadrupedEnv({})
# from ray.rllib.models.preprocessors import get_preprocessor
# prep = get_preprocessor(testenv.observation_space)
# print(prep)
# print(testenv.reset().shape)
# <class 'ray.rllib.models.preprocessors.NoPreprocessor'>
# (48,)

"""
Notes:

seems simple_optimizer should be False

# most recent test was 128
"""
NUM_CPUS = 8

num_samples_each_worker = int(4096 / NUM_CPUS)
# num_samples_each_worker = int(1024 / NUM_CPUS)


config_PPO={"env": env, #DummyQuadrupedEnv,
            "num_gpus": 0,
            "num_workers": NUM_CPUS,
            "num_envs_per_worker": 1, # to test if same as SB
            "lr": 1e-4,
            "monitor": True,
            "model": model,
            "train_batch_size": NUM_CPUS*num_samples_each_worker, #4096, #n_steps,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,#256,# WAS 64#nminibatches, ty 256 after
            "rollout_fragment_length": num_samples_each_worker,#1024,
            "clip_param": 0.2,
            #"use_kl_loss": False,
            "vf_clip_param": 1,#0.2, #0.2,#10, # in SB, this is default as the same as policy clip_param
            "vf_loss_coeff": 0.5,
            "lambda":0.95, #lam in SB
            "grad_clip": 0.5, # max_grad_norm in SB
            "kl_coeff": 0.0, # no KL in SB, look into this more
            "kl_target": 0.01,
            "entropy_coeff": 0.0,
            "observation_filter": "MeanStdFilter", # THIS IS NEEDED IF NOT USING VEC_NORMALIZE
            "clip_actions": True,
            "vf_share_layers": False, # this is true in SB, # Better to not have these shared
            "normalize_actions": True, # added
            "preprocessor_pref": "rllib", # what are these even doing? anything?
            "simple_optimizer": False, # test this next
            #"batch_mode": "complete_episodes", #"truncate_episodes", 
            "batch_mode": "truncate_episodes", 
            #"custom_preprocessor": "NoPreprocessor"

             
            # "horizon": 1001,
            # this seemed to have an effect
            # "no_done_at_end": True,
            #################3 change
            "no_done_at_end": False,
            # "soft_horizon": True,
            # "horizon": None, # changing from 501
            "shuffle_buffer_size": NUM_CPUS*num_samples_each_worker, #4096,

        }
config.update(config_PPO)
for k,v in config.items():
    print(k,v)
#sys.exit()

# from ray.tune.logger import pretty_print
# # import pdb
# # pdb.set_trace()
# trainer = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)

# for i in range(int(10000000 / 4096)):
#    # Perform one iteration of training the policy with PPO
#    result = trainer.train()
#    print(pretty_print(result))
#    if i % 10 == 0:
#        checkpoint = trainer.save()
#        print("checkpoint saved at", checkpoint)

tune.run("PPO",config=config, 
                local_dir=SAVE_RESULTS_PATH,#'./ray_results/', 
                checkpoint_freq=20, 
                verbose=2,
                stop= {"timesteps_total": 100000000}
                )


#############################################################
# Old tests
#############################################################
#tune.run(PPOTrainer, config={"env": QuadrupedGymEnv}) ]
# trainer = ppo.PPOTrainer(env="my_env")
# while True:
#     print(trainer.train())