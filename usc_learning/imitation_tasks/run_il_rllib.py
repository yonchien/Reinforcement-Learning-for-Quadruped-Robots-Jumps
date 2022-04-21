import os
from datetime import datetime
import numpy as np
import gym
# ray
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.rllib.agents import ppo, sac, ddpg, ars, es
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
# quadruped env
# from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
from usc_learning.imitation_tasks.imitation_gym_env_lstm import ImitationGymEnv
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


recent run:
imitation_tasks/ray_results/030221015318/SAC

"""
monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'


ALG = "SAC"
# ALG = "PPO"
USE_IMITATION_ENV = True

USE_LSTM = False


# use VecNormalize
USING_VEC_ENV = False
DONT_NORM_BOOLS = False
SAVE_RESULTS_PATH = os.path.join('./ray_results', monitor_dir)

# get absolute path to the current directory for saving
script_dir = os.path.dirname(os.path.abspath(__file__))

# for copying all relevant files and VecNormalize params (running mean/std)
save_path = os.path.join(script_dir, SAVE_RESULTS_PATH)
vec_stats = os.path.join(save_path, "vec_normalize.pkl")
os.makedirs(save_path, exist_ok=True)
# copy relevant files to save directory to provide snapshot of code changes
copy_files(save_path)


class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, env_config):
        #env = QuadrupedGymEnv(render=False,hard_reset=True)
        # if USING_VEC_ENV:
        #     # same as in stable baselines
        #     if USE_IMITATION_ENV:
        #         env = lambda: ImitationGymEnv()
        #     else:
        #         env = lambda: QuadrupedGymEnv(render=False)
        #     env = make_vec_env(env, monitor_dir=save_path)
        #     #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)
        #     env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=1, clip_obs=100.)
        #     #env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)
        #     self.env = env 
        #     self._save_vec_stats_counter = 0
        #     # write env configs to reload (i.e if don't need to inspect whole code base)
        #     write_env_config(save_path,env)
        # else:
        self.env = ImitationGymEnv(observation_space_mode="MOTION_IMITATION_EXTENDED2_CONTACTS_HISTORY")
        write_env_config(save_path, self.env)
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

        # print("OBS_SIZE:", obs.shape)
        #print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            #sys.exit()

        print("OBS_SIZE:", obs.shape)
        return obs, rew, done, info


ray.init()

from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_action_dist("SquashedGaussian", SquashedGaussian)

# Neural Network parameters
model = {"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh"}

# test to see if we get same thing as in stable baselines
NUM_CPUS = 10
n_steps = 4 * 1024
# should this really be the ENTIRE batch?
nminibatches = 256 #16 #256  #int(n_steps / 64)
num_samples_each_worker = 1024 #512 #1024

# timesteps_per_iteration = 100
# checkpoint_freq = 10
timesteps_per_iteration = 10
checkpoint_freq = 10
if timesteps_per_iteration > 101:
    checkpoint_freq = 1
else:
    checkpoint_freq = 10

env = DummyQuadrupedEnv


if ALG == "SAC":
    config = sac.DEFAULT_CONFIG.copy()

    # test 2 - like in SB
    config_SAC = { 
        "env": DummyQuadrupedEnv,
        #"horizon": 1000,
        "soft_horizon": False,
        "Q_model": {
          "fcnet_activation": "tanh",
          "fcnet_hiddens": model["fcnet_hiddens"],
          },
        "policy_model": {
          "fcnet_activation": "tanh",
          "fcnet_hiddens":  model["fcnet_hiddens"],
          },
        "tau": 0.005,
        # Initial value to use for the entropy weight alpha.
        #"initial_alpha": 1.0,
        "target_entropy": "auto",
        "no_done_at_end": False,
        "n_step": 1,
        "rollout_fragment_length": 1,
        "prioritized_replay": False,#,True,
        "train_batch_size": 64,#100,#256,
        "target_network_update_freq": 1,#1,
        "observation_filter": "MeanStdFilter", # THIS IS NEEDED IF NOT USING VEC_NORMALIZE
        # not in original 
        #"grad_clip": 0.5,
        "timesteps_per_iteration": timesteps_per_iteration,
        "buffer_size": 100000,
        "learning_starts": 100,
        "optimization": {
          "actor_learning_rate": 0.0003,
          "critic_learning_rate": 0.0003,
          "entropy_learning_rate": 0.0003,
            # "actor_learning_rate": 0.0001,
            # "critic_learning_rate": 0.0001,
            # "entropy_learning_rate": 0.0001,
          },
        "num_workers": NUM_CPUS,
        "num_gpus": 1,
        # should not clip actions? - in defaults is False
        "clip_actions": True,
        "normalize_actions": True,
        "preprocessor_pref": "rllib", # what are these even doing? anything?
        # "evaluation_interval": 1,
        # "metrics_smoothing_episodes": 5,
        }
    config.update(config_SAC)
    # import pdb
    # pdb.set_trace()
    tune.run("SAC", config=config,#config_SAC,
                    local_dir=SAVE_RESULTS_PATH,
                    checkpoint_freq=checkpoint_freq,
                    verbose=2,
                    stop= {"timesteps_total": 10} # 10000000
                    )
    ray.shutdown()
    #sys.exit()

reporter = CLIReporter(max_progress_rows=10)

if ALG == "PPO":

    NUM_CPUS = 15
    num_samples_each_worker = int(6000 / NUM_CPUS)
    train_batch_size = NUM_CPUS * num_samples_each_worker

    config = ppo.DEFAULT_CONFIG.copy()

    config_PPO = {"env": DummyQuadrupedEnv,
                  "num_gpus": 1,
                  "num_workers": NUM_CPUS,
                  "num_envs_per_worker": 1,  # to test if same as SB
                  "lr": 1e-4,
                  # "monitor": True,
                  "model": model,
                  "train_batch_size": train_batch_size,  # 4096, #n_steps,
                  "num_sgd_iter": 10,
                  "sgd_minibatch_size": 256,  # 256,# WAS 64#nminibatches, ty 256 after
                  "rollout_fragment_length": 400,  # 1024,
                  "clip_param": 0.2,
                  # "use_kl_loss": False,
                  "vf_clip_param": 1,  # 0.2,#10, # in SB, this is default as the same as policy clip_param
                  "vf_loss_coeff": 0.5,
                  "lambda": 0.95,  # lam in SB
                  "grad_clip": 0.5,  # max_grad_norm in SB
                  "kl_coeff": 0.2,  # no KL in SB, look into this more
                  "kl_target": 0.01,
                  "entropy_coeff": 0.0,
                  "observation_filter": "MeanStdFilter",  # THIS IS NEEDED IF NOT USING VEC_NORMALIZE
                  # "observation_filter": PartialFilter2,
                  "clip_actions": True,
                  # "vf_share_layers": True,  # this is true in SB # TODO: test this, tune the vf_clip_param
                  "normalize_actions": True,  # added
                  "preprocessor_pref": "rllib",  # what are these even doing? anything?
                  # "batch_mode": "complete_episodes", #"truncate_episodes",
                  "batch_mode": "truncate_episodes",
                  # "custom_preprocessor": "NoPreprocessor"

                  # this seemed to have an effect
                  "no_done_at_end": True,
                  # "soft_horizon": True,
                  # "horizon": None, # changing from 501
                  "shuffle_buffer_size": NUM_CPUS * num_samples_each_worker,  # 4096,
                  # "callbacks": {
                  #     "on_train_result": on_train_result,
                  # },
                  "framework": "tf",
                  "eager_tracing": True,
                  # "gamma": 0.95,
                  }

    config.update(config_PPO)
    tune.run("PPO", config=config,
             local_dir=SAVE_RESULTS_PATH,
             checkpoint_freq=50,
             verbose=2,
             checkpoint_at_end=True,
             # stop=stopper,
             stop={"timesteps_total": 2},
             progress_reporter=reporter
             )
