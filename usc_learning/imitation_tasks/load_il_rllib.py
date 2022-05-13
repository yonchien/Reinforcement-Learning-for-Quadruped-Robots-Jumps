# MLP

#!/usr/bin/env python
"""Load rllib. Pretty hacky. """
import argparse
import collections
import copy
import json
import os, sys
from pathlib import Path
import pickle5 as pickle
import shelve
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import gym
import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env
from ray.rllib.agents import ppo, sac, ddpg, es, ars


from usc_learning.utils.utils import plot_results, load_rllib, load_rllib_v2
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
import numpy as np
# stable baselines vec env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from usc_learning.utils.file_utils import get_latest_model_rllib, get_latest_directory, read_env_config

from ray.rllib.models import ModelCatalog
from usc_learning.learning.rllib.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
ModelCatalog.register_custom_model("my_fcnet", MyFullyConnectedNetwork)
################################### Notes

# TODO: add option to load different trained algs, this assumes PPO only so far
def get_log_dir(dir_name):
    return os.path.join('./ray_results', dir_name)


log_dir = '030221015318'
log_dir = '030421231941'
log_dir = '030421201627'
# log_dir = '030521003015'
# log_dir = '030421190920'
# not in consideration:
# log_dir = '030521053657' # min mu 1 (stoppped), not good
# log_dir = '030521030604' # 0.08, 0.5 not  far enough, stopped

# all currently running with various coefficients of friction and heights 
log_dir = '030521023907' # 0.05, 0.7  not good transfer 
# log_dir = '030521023930' # 0.05, 0.5, bad transfer 
# log_dir = '030521024013' # 0.08, 0.7, close, close with front block
# log_dir = '030521030325' # 0.1, 1, distance, but turns , still turns midair (got a good video for the paper)

# height, mu
# log_dir = '030521053601' # 0.1, 0.5, transfer decent 
# log_dir = '030521053628' # 0.1, 0.8  transfer not good , front block forward but sidways
# log_dir = '030521053731' # 0.08, 1   transfer close, but turns, 2nd try not good 
# log_dir = '030521055234' # 0.05, 0.6 transfer not really

log_dir = '030521183527' # with noise

# post submission, verify ranges 
log_dir= '030621120331' # jump and roll 
log_dir = '030621120359'
log_dir = '030621120650'
log_dir ='030821121850' # started later

# CHUONG TRIED
log_dir ='042122173936' # chuong- LSTM-PPO
log_dir ='042622140629' # chuong -MLP-PPO, "fcnet_hiddens": [200, 100]
# log_dir ='042922102335' # some steps
log_dir ='042922103532' # SAC+ change action repeat to 30, 2 million training steps

log_dir ='050122161428' # SAC+ change action repeat to 50, 350000 training steps, reward threshold=0.5
log_dir ='050122224217' # SAC+ change action repeat to 50, 350000 training steps, reward threshold=0.05
log_dir ='050222234904' # SAC+ change action repeat to 50, 350000 training steps, reward threshold=0.05
log_dir ='050322095420' # SAC+ change action repeat to 50, 350000 training steps, reward threshold=0.5
log_dir ='050322162428' # SAC+ change action repeat to 50, 350000 training steps, reward threshold=0.05, action dim=12*2, partial traj
log_dir ='050422182630'  # GUILAUME
################################################################################################
### Script - shouldn't need to change anything below unless trying to load a specific checkpoint
#################################################################################################
USING_VEC_ENV = False
USE_ROS = False

log_dir = get_log_dir(log_dir)
alg_dir = get_latest_directory(log_dir) #i.e. {log_dir}/PPO
train_dir = get_latest_directory(alg_dir) # i.e. {log_dir}/PPO/PPO_DummyQuadrupedEnv_0_2020-07-11_13-27-26rtt2kwpd
print('log dir', log_dir)
print('alg dir', alg_dir, alg_dir[-3:])
print('train dir', train_dir)

ALG = alg_dir[-3:]
if '/' in ALG:
    ALG = ALG[1:]
# sys.exit()

# most recent checkpoint in train_dir
checkpoint = get_latest_model_rllib(train_dir)
data = load_rllib(train_dir)

try:
    sys.path.append(log_dir[:-log_idx])
except:
    print("COULDN'T LOAD")
    pass
stats_path = str(log_dir)
print('stats_path', stats_path)
vec_stats_path = os.path.join(stats_path, "vec_normalize.pkl")


# get env config if available
try:
    # stats path has all saved files (quadruped, configs_a1, as well as vec params)
    env_config = read_env_config(stats_path)
    #sys.path.append(stats_path)
    sys.path.insert(0,stats_path)

    # from imitation_gym_env_lstm import ImitationGymEnv
    # from quadruped import Quadruped
    # # added 
    # import quadruped_gym_env
    
    # check env_configs.txt file to see whether to load aliengo or a1
    with open(stats_path + '/env_configs.txt') as configs_txt:
        tmp = configs_txt.read()
        print('configs txt',tmp)
        print('testing:', 'laikago' in tmp)
        #sys.exit()
        if 'aliengo' in configs_txt.read():
            print('IMPORTING ALIENGO')
            import configs_aliengo as robot_config
        elif 'laikago' in tmp: #configs_txt.read():
            print('IMPORTING LAIKAGO')
            import configs_laikago as robot_config
        else:
            print('IMPORTING A1')
            import configs_a1 as robot_config

    env_config['robot_config'] = robot_config
except:
    print('*'*50,'could not load env stats','*'*50)
    env_config = {}
    sys.exit()
if not USE_ROS:
    env_config['render'] = True
# env_config['record_video'] = True
# env_config['randomize_dynamics'] = False
# env_config['add_terrain_noise'] = True
# env_config['test_mass'] = [10, 0.0, 0, 0.05]

env_config['land_and_settle'] = True
env_config['test_heights'] = np.array([ [0.01, 0.05, 0.01, 0.1] ,
                                   [0.05, 0.01, 0.1 , 0.01]]).T

for k,v in env_config.items():
    print(k,v)

print("kpCartesian", robot_config.kpCartesian)
print("kdCartesian", robot_config.kdCartesian)
print("kp joint", robot_config.MOTOR_KP)
print("kd joint", robot_config.MOTOR_KD)

class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, dummy_env_config):
        self.env = ImitationGymEnv(**env_config)
        #write_env_config(save_path,self.env)
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


################################### register env
# needed for this file
register_env("quadruped_env",  lambda _: DummyQuadrupedEnv(_))
register_env("quadruped_gym_env",  lambda _: DummyQuadrupedEnv(_))

# load training configurations (can be seen in params.json)
config_path = os.path.join(train_dir, "params.pkl")

with open(config_path, "rb") as f:
    config = pickle.load(f)

config["num_workers"] = 0
# ray.init()#memory=1024)
# ray.init(memory=52428800, object_store_memory=78643200)

if ALG=="PPO":
    # agent = ppo.PPOTrainer(config=config, env="quadruped_env")
    agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)
elif ALG=="SAC":
    print('load SAC')
    agent = sac.SACTrainer(config=config, env=DummyQuadrupedEnv)
elif ALG=="TD3":
    agent = ddpg.td3.TD3Trainer(config=config, env=DummyQuadrupedEnv)
elif ALG=="ARS":
    config["num_workers"] = 0
    # agent = ars.ARSTrainer(config=config, env=DummyQuadrupedEnv)
    cls = get_trainable_cls("ARS")
    agent = cls(env="quadruped_env", config=config)
elif ALG=="ES":
    config["num_workers"] = 3
    agent = es.ESTrainer(config=config, env=DummyQuadrupedEnv)
else:
    raise ValueError('not a valid alg')
# Load state from checkpoint.
print('checkpoint', checkpoint)
agent.restore(checkpoint)

env = agent.workers.local_worker().env
print(env)
print('agent',agent)

# import pdb
# pdb.set_trace()
obs = env.reset()
# action = agent.compute_action(obs)

write_out_path_root = '/home/guillaume/catkin_ws/src/USC-AlienGo-Software/laikago_ros/jumping_iros/'
# env.envs[0].env._traj_task.writeoutTraj(write_out_path)
write_out_path = write_out_path_root + 'model_imitation/'

TEST_STEPS = 10
# USE_ROS = False
if USE_ROS:
    from ray.tune.trial import ExportFormat
    import tensorflow as tf
    policy = agent.get_policy()
    print(policy)
    print(policy.get_session())
    print('variables', policy.variables())
    i = tf.initializers.global_variables()
    with policy.get_session() as sess:
        saver = tf.train.Saver(policy.variables())#tf.global_variables())
        saver.save(sess, write_out_path + 'model')
        tf.train.write_graph(sess.graph, '.', write_out_path + 'graph.pb', as_text=False)
    


    # save observation and action space 
    rllib_mean_std_filter = agent.workers.local_worker().filters['default_policy']
    try:
        var = np.square(rllib_mean_std_filter.rs.std)
        vecnorm_arr = np.vstack((rllib_mean_std_filter.rs.mean, 
                                 rllib_mean_std_filter.rs.var,
                                env.env.observation_space.high,
                                env.env.observation_space.low))

        # quadruped_env = env.env.env.env


        np.savetxt(write_out_path + 'vecnorm_params.csv',vecnorm_arr,delimiter=',')
    except:
        print('Not using meanstdfilter in training, so didnt save')
    ray.shutdown()
    sys.exit()

joint_taus = np.zeros((12,TEST_STEPS))
joint_vels = np.zeros((12,TEST_STEPS))
joint_pos = np.zeros((12,TEST_STEPS))
episode_reward = 0
num_steps = 0
# velocities
des_vels = []
act_vels = []

for _ in range(TEST_STEPS):
    #print(obs)
    if USE_ROS:
        # query service
        rl_obs = RLObs()
        rl_obs.data = obs
        rl_act = tf_comm(rl_obs)
        action = np.clip(rl_act.rlAct.data, -1, 1)
    else:
        action = agent.compute_action(obs, explore=False) # doesn't really matter?
        #print(agent.compute_action([0]*48))
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    num_steps += 1
    if done:
        print('episode reward:', episode_reward, "num_steps:", num_steps)
        #print(env.env.env.env.venv.envs[0].env._robot.GetBasePosition() )
        print(info)
        episode_reward = 0
        num_steps = 0
        obs = env.reset()
    #time.sleep(0.01)
    # print(obs)

    # des_vels.append(env._get_des_vel())
    # act_vels.append(env._robot.GetBaseLinearVelocity()[0])

    # joint_taus[:,_] = env.env.env.env._robot.GetMotorTorques()
    # joint_vels[:,_] = env.env.env.env._robot.GetMotorVelocities()
    # joint_pos[:,_] = env.env.env.env._robot.GetMotorAngles()