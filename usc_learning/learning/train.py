#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 02:12:46 2020

@author: shashank
"""
import sys
import os
from os import mkdir
from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, DDPG, PPO1
import tensorflow as tf
import gym
import numpy as np
from stable_baselines.common.callbacks import EvalCallback, EventCallback, BaseCallback, CheckpointCallback
from datetime import datetime
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

BASE_LOG_LOCATION = "logs2"




def make_env():
    # returns a function which creates a single environment
    def _thunk():
        env = AlienGoGymEnv(isrender = False, control_mode="FORCE")
        return env
    return _thunk


def get_train_vec_env(N = 8):
    return SubprocVecEnv([make_env() for _ in range(N)])

def get_train_env():
    return AlienGoGymEnv(isrender = False, control_mode="FORCE")

def get_test_env():
    return AlienGoGymEnv(isrender = True, control_mode="FORCE")

def get_env_config():
    env = get_train_env()
    env_config = {"control_mode":env._control_mode,
                    "task_env":env._TASK_ENV,
                    "actionRepeat":env._action_repeat,
                    "timeStep":env._time_step,
                    "env2d":env._env2d,

                    "moving_distance_reward":env._relZdist_w,
                    "abs_distance_reward":env._absZdist_w,
                    "orientation_reward":env._ori_w,
                    "shake_reward":env._shake_w,
                    "shake_reward_for_velocity":env._shake_vel_w,
                    "shake_reward_for_orientation":env._shake_orn_w,
                    "energy_reward":env._energy_w,

                    "tipping_threshold":env._tip_threshold,
                    "termination_penalty":env._termination_w,

                    "joint_kp_gains":env._robot._motor_kps,
                    "joint_kd_gains":env._robot._motor_kds                    
                    }
    env.close()
    return env_config


class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every `n_steps` timesteps

    :param n_steps: (int) Number of timesteps between two trigger.
    :param callback: (BaseCallback) Callback that will be called
        when the event is triggered.
    """
    def __init__(self, env, n_steps: int, path_to_videos:str):
        super().__init__()
        self.t_env = env
        self.n_steps = n_steps
        self.last_time_trigger = 0
        self.path_to_videos = path_to_videos

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True

    def _on_event(self):
        obs = self.t_env.reset()
        path_to_video = os.path.join(self.path_to_videos,str(self.num_timesteps))
        self.t_env._robot.startRecordingVideo(path_to_video+".mp4")
        tot_rew = 0
        for _ in range(200):
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.t_env.step(action)
            # self.t_env.render()
            tot_rew += rewards
            if done:
                obs = self.t_env.reset()
        self.t_env._robot.stopRecordingVideo()
        print(tot_rew)


def train_ddpg(steps = 1000000):
    ddpg_cofig = {"gamma":0.99, "memory_policy":None, "nb_train_steps":50,"eval_env":None,
                    "nb_rollout_steps":100, "nb_eval_steps":100, "param_noise":None, "action_noise":None,
                    "normalize_observations":False, "tau":0.001, "batch_size":128, 
                    "param_noise_adaption_interval":50, "normalize_returns":False, "enable_popart":False, 
                    "observation_range":(-5., 5.), "critic_l2_reg":0., "return_range":(-np.inf, np.inf), 
                    "actor_lr":1e-4, "critic_lr":1e-3, "clip_norm":None, "reward_scale":1.,
                    "render":False, "render_eval":False, "memory_limit":None, "buffer_size":500000, 
                    "random_exploration":0.0, "verbose":1, "tensorboard_log":"logs/ddpg/", 
                    "_init_setup_model":True, "policy_kwargs":None,
                    "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":1}



    test_env = get_test_env()
    train_env = get_train_env()


    eval_callback = EveryNTimesteps(test_env,500)

    model = DDPG(MlpPolicy, env = train_env, **ddpg_cofig)

    model.learn(total_timesteps=500000, callback=eval_callback)
    model.save("models/ddpg/"+name)


def train_ppo2(steps = 1000000):
    ppo2_config = {'gamma': 0.99, 'n_steps': 512, 'ent_coef': 0.01, 
    'learning_rate': 0.0025, 'vf_coef': 0.5, 'max_grad_norm': 0.5, 
    'lam': 0.95, 'nminibatches': 2, 'noptepochs': 4, 'cliprange': 0.2, 
    'cliprange_vf': None, 'verbose': 0, 'tensorboard_log': BASE_LOG_LOCATION+'/ppo2/tensorboard/', 
    '_init_setup_model': True, 'policy_kwargs': None, 'full_tensorboard_log': False, 
    'seed': None, 'n_cpu_tf_sess': None}

    env_config = get_env_config()

    os.makedirs(BASE_LOG_LOCATION, exist_ok=True)
    os.makedirs(os.path.join(BASE_LOG_LOCATION,"ppo2"), exist_ok=True)

    path_to_current_log = os.path.join(*[BASE_LOG_LOCATION,"ppo2",datetime.now().strftime("%m%d%y%H%M%S")])

    path_to_models = os.path.join(path_to_current_log,"model_ckpts")
    path_to_videos = os.path.join(path_to_current_log,"videos")
    os.makedirs(path_to_videos, exist_ok=True)

    f = open(os.path.join(path_to_current_log,"configs.txt"),"w")
    f.write("ppo2 config:")
    f.write(str(ppo2_config))
    f.write("\n")
    f.write("env config:")
    f.write(str(env_config))
    f.close()
    test_env = get_test_env()
    train_env = VecNormalize(get_train_vec_env())

    eval_callback = EveryNTimesteps(test_env,50000,path_to_videos)
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=path_to_models,
                                         name_prefix='aliengo_ppo2')

    model = PPO2(MlpPolicy, env = train_env, **ppo2_config)

    model.learn(total_timesteps=steps, callback=[eval_callback,checkpoint_callback])
    model.save(path_to_models+"/final_model")

def train_ppo(steps = 1000000):
    ppo_config = {"gamma":0.99, "timesteps_per_actorbatch":256, "clip_param":0.2, "entcoeff":0.01,
                 "optim_epochs":4, "optim_stepsize":1e-3, "optim_batchsize":64, "lam":0.95, "adam_epsilon":1e-5,
                 "schedule":'linear', "verbose":0, "tensorboard_log":"logs/ppo/tensorboard/", "_init_setup_model":True,
                 "policy_kwargs":None, "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":1}

    os.makedirs(BASE_LOG_LOCATION, exist_ok=True)
    os.makedirs(os.path.join(BASE_LOG_LOCATION,"ppo"), exist_ok=True)

    path_to_current_log = os.path.join(*[BASE_LOG_LOCATION,"ppo",datetime.now().strftime("%m%d%y%H%M%S")])

    path_to_models = os.path.join(path_to_current_log,"model_ckpts")
    path_to_videos = os.path.join(path_to_current_log,"videos")
    os.makedirs(path_to_videos, exist_ok=True)

    f = open(os.path.join(path_to_current_log,"configs.txt"),"w")
    f.write( str(ppo_config) )
    f.close()
    test_env = get_test_env()
    train_env = get_train_env()

    eval_callback = EveryNTimesteps(test_env,500000,path_to_videos)
    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=path_to_models,
                                         name_prefix='aliengo_ppo')

    model = PPO1(LstmPolicy, env = train_env, **ppo_config)

    model.learn(total_timesteps=steps, callback=[eval_callback,checkpoint_callback])
    model.save(path_to_models+"/final_model")

if __name__ == "__main__":
    train_ppo2(50000000)
    # config = get_env_config()
    # print(config)