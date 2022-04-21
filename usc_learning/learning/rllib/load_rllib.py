#!/usr/bin/env python
"""Load rllib. Pretty hacky. """
import argparse
import collections
import copy
import json
import os, sys
from pathlib import Path
import pickle
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
import numpy as np
# stable baselines vec env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from usc_learning.utils.file_utils import get_latest_model_rllib, get_latest_directory, read_env_config

from ray.rllib.models import ModelCatalog
from usc_learning.learning.rllib.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
ModelCatalog.register_custom_model("my_fcnet", MyFullyConnectedNetwork)
################################### Notes

def get_log_dir(dir_name):
    return os.path.join('./ray_results', dir_name)
#################################################################################################
### CONFIGS
#################################################################################################


# variable impedance gains
log_dir = '032221125756'
# log_dir = '032221125835' # rough terrain
log_dir = '032221153058' # rough terrain 8 kg mass
log_dir = '032621210931'
log_dir = '032621224313'
log_dir = '032721001049'

# sys.path.insert(0,'../../envs/quadruped_master/')
################################################################################################
### Script - shouldn't need to change anything below unless trying to load a specific checkpoint
#################################################################################################
USING_VEC_ENV = False
USE_ROS = False
WRITE_OUT_POLICY = False

log_dir = get_log_dir(log_dir)
alg_dir = get_latest_directory(log_dir) #i.e. {log_dir}/PPO
train_dir = get_latest_directory(alg_dir) # i.e. {log_dir}/PPO/PPO_DummyQuadrupedEnv_0_2020-07-11_13-27-26rtt2kwpd
print('log dir', log_dir)
print('alg dir', alg_dir, alg_dir[-3:])
print('train dir', train_dir)

ALG = alg_dir[-3:]
if '/' in ALG:
    ALG = ALG[1:]

# most recent checkpoint in train_dir
checkpoint = get_latest_model_rllib(train_dir)

#checkpoint = './ray_results/100520084804/PPO/PPO_DummyQuadrupedEnv_0_2020-10-05_08-48-06q1thqse2/checkpoint_500/checkpoint-500'
# visualize episode lengths/rewards over training
data = load_rllib(train_dir)


# try:
#     sys.path.append(log_dir[:-log_idx])
#     #sys.path.insert(0,log_dir[:-log_idx])
#     #print(sys.path)
#     #from quadruped_gym_env import QuadrupedGymEnv
# except:
#     print("COULDN'T LOAD")
#     pass
stats_path = str(log_dir)
print('stats_path', stats_path)
vec_stats_path = os.path.join(stats_path,"vec_normalize.pkl")


# get env config if available
try:
    # stats path has all saved files (quadruped, configs_a1, as well as vec params)
    env_config = read_env_config(stats_path)
    #sys.path.append(stats_path)

    sys.path.insert(0,stats_path)

    # from quadruped_gym_env import QuadrupedGymEnv
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
env_config['render'] = True
# env_config['record_video'] = True
env_config['randomize_dynamics'] = False
# env_config['add_terrain_noise'] = True
# env_config['test_mass'] = [10, 0.0, 0, 0.05]

for k,v in env_config.items():
    print(k,v)

print("kpCartesian", robot_config.kpCartesian)
print("kdCartesian", robot_config.kdCartesian)
print("kp joint", robot_config.MOTOR_KP)
print("kd joint", robot_config.MOTOR_KD)
# sys.exit()
################################### Env

class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, dummy_env_config):
        # set up like stable baselines (will this work?)
        #env = QuadrupedGymEnv(render=False,hard_reset=True)
        if USING_VEC_ENV:
            env = lambda: QuadrupedGymEnv(**env_config) #render=True,hard_reset=True
            env = make_vec_env(env, n_envs=1)
            env = VecNormalize.load(vec_stats_path, env)
            # do not update stats at test time
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False

            self.env = env 
            self._save_vec_stats_counter = 0
        else:
            self.env = QuadrupedGymEnv(**env_config)#render=True,hard_reset=True)
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
            # no need to save anything at load time
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        """step env """
        if USING_VEC_ENV:
            obs, rew, done, info = self.env.step([action])
            obs = obs[0]
            rew = rew[0]
            done = done[0]
            info = info[0]
        else:
            obs, rew, done, info = self.env.step(action)
        #print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            sys.exit()
        return obs, rew, done, info 
#     # def render(self,):
#     #     self.env.render()

################################### register env
# needed for this file
register_env("quadruped_env",  lambda _: DummyQuadrupedEnv(_))
register_env("quadruped_gym_env",  lambda _: DummyQuadrupedEnv(_))

# load training configurations (can be seen in params.json)
config_path = os.path.join(train_dir, "params.pkl")

with open(config_path, "rb") as f:
    config = pickle.load(f)

# print('*'*80)
# print('config',config)
# for k,v in config.items():
#     print(k,v)
# print('*'*80)
# complains if 1.. but is should be 1?
config["num_workers"] = 0

ray.init()#memory=1024)
# Create the Trainer from config.
#cls = get_trainable_cls(alg)
#print('cls',cls)
#agent = cls(env="quadruped_env", config=config)
#print('agent',agent)
#env = DummyQuadrupedEnv('fake')

if ALG=="PPO":
    # agent = ppo.PPOTrainer(config=config, env="quadruped_env")
    agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)
elif ALG=="SAC":
    agent = sac.SACTrainer(config=config, env=DummyQuadrupedEnv)
elif ALG=="TD3":
    agent = ddpg.td3.TD3Trainer(config=config, env=DummyQuadrupedEnv)
elif ALG=="ARS":
    print('ARS not working.. worker issue')
    config["num_workers"] = 0
    # agent = ars.ARSTrainer(config=config, env=DummyQuadrupedEnv)
    cls = get_trainable_cls("ARS")
    agent = cls(env="quadruped_env", config=config)
elif ALG=="ES":
    print('ES not working.. worker issue')
    config["num_workers"] = 3
    agent = es.ESTrainer(config=config, env=DummyQuadrupedEnv)
else:
    raise ValueError('not a valid alg')

# Load state from checkpoint.
print('restore checkpoint', checkpoint)
agent.restore(checkpoint)
env = agent.workers.local_worker().env
print('env', env)
print('agent',agent)

# import pdb
# pdb.set_trace()
obs = env.reset()
action = agent.compute_action(obs)
#sys.exit()
# if 1:
#     #env = gym.make("quadruped_env")
#     env = DummyQuadrupedEnv('fake')
#     policy_map = {DEFAULT_POLICY_ID: agent.policy}
# else:
#     env = agent.workers.local_worker().env
#     policy_map = agent.workers.local_worker().policy_map
#     state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
#     use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
#import csv
# with open('./exported/vecnorm_params.csv', 'wb') as f:
#     writer = csv.writer(f, delimiter=',')


# write out in general?
write_out_path = './model/'

if WRITE_OUT_POLICY:
    from ray.tune.trial import ExportFormat
    import tensorflow as tf

    policy = agent.get_policy()
    print(policy)
    print(policy.get_session())
    print('variables', policy.variables())
    i = tf.initializers.global_variables()

    #saver = tf.train.Saver(tf.all_variables())
    with policy.get_session() as sess:
        saver = tf.train.Saver(policy.variables())#tf.global_variables())
        # saver.save(sess, './exported/my_model')
        # tf.train.write_graph(sess.graph, '.', './exported/graph.pb', as_text=False)
        saver.save(sess, write_out_path + 'my_model')
        tf.train.write_graph(sess.graph, '.', write_out_path + 'graph.pb', as_text=False)


    # if NOT vecnormalized, need to save normalization parameters based on MeanStdFilter 
    if not USING_VEC_ENV:
        rllib_mean_std_filter = agent.workers.local_worker().filters['default_policy']
        var = np.square(rllib_mean_std_filter.rs.std)
        vecnorm_arr = np.vstack((rllib_mean_std_filter.rs.mean, 
                                 rllib_mean_std_filter.rs.var,
                                env.env.observation_space.high,
                                env.env.observation_space.low))

        quadruped_env = env.env.env.env
    else:
        # save observation and action space 
        venv = env.env.env.env
        quadruped_env = env.env.env.env.venv.envs[0].env
        print('mean', venv.obs_rms.mean)
        print('var ', venv.obs_rms.var)
        vecnorm_arr = np.vstack((venv.obs_rms.mean, 
                                venv.obs_rms.var,
                                venv.observation_space.high,
                                venv.observation_space.low))
    
    if env_config['motor_control_mode'] == "IMPEDANCE":
        act_space_arr = np.vstack((
                            np.concatenate((quadruped_env._robot_config.IK_POS_UPP_LEG,quadruped_env._robot_config.MAX_NORMAL_FORCE)),
                            np.concatenate((quadruped_env._robot_config.IK_POS_LOW_LEG,quadruped_env._robot_config.MIN_NORMAL_FORCE))
                            ))
    elif env_config['motor_control_mode'] == 'IMPEDANCE_POS_ONLY':
        act_space_arr = np.vstack((
                            quadruped_env._robot_config.IK_POS_UPP_LEG,
                            quadruped_env._robot_config.IK_POS_LOW_LEG
                            ))
    elif env_config['motor_control_mode'] == 'PD':
        act_space_arr = np.vstack((
                            quadruped_env._robot_config.UPPER_ANGLE_JOINT,
                            quadruped_env._robot_config.LOWER_ANGLE_JOINT,
                            ))
        # also write out gains 
        motor_gains = np.array([quadruped_env._robot_config.MOTOR_KP[0], quadruped_env._robot_config.MOTOR_KD[0]])
        print('writing out motor gains:', motor_gains )
        np.savetxt(write_out_path + 'joint_gains.csv', motor_gains,delimiter=',')
    else:
        raise ValueError('add action space scaling ... ')

    # save normaliziation parameters and action space range
    np.savetxt(write_out_path + 'vecnorm_params.csv',vecnorm_arr,delimiter=',')
    np.savetxt(write_out_path + 'action_space.csv', act_space_arr, delimiter=',')

    ray.shutdown()
    sys.exit()

USE_ROS = False
if USE_ROS:
    import rospy
    from laikago_msgs.srv import TFService, TFServiceRequest, TFServiceResponse
    from laikago_msgs.msg import RLObs
    rospy.init_node('pyb_interface', anonymous=True)

    rospy.wait_for_service('/laikago_gazebo/tf_comm')
    try:
        tf_comm = rospy.ServiceProxy('/laikago_gazebo/tf_comm', TFService,persistent=True)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

#obs = env.reset()
episode_reward = 0
num_steps = 0
# velocities
des_vels = []
act_vels = []

TEST_STEPS = 1002
joint_taus = np.zeros((12,TEST_STEPS))
joint_vels = np.zeros((12,TEST_STEPS))
joint_pos = np.zeros((12,TEST_STEPS))

for _ in range(TEST_STEPS):
    #print(obs)
    if USE_ROS:
        # query service
        rl_obs = RLObs()
        rl_obs.data = obs
        rl_act = tf_comm(rl_obs)
        action = np.clip(rl_act.rlAct.data, -1, 1)
    else:
        action = agent.compute_action(obs, explore=True) # doesn't really matter?
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

    des_vels.append(env.env.env.env._get_des_vel())
    act_vels.append(env.env.env.env._robot.GetBaseLinearVelocity()[0])

    joint_taus[:,_] = env.env.env.env._robot.GetMotorTorques()
    joint_vels[:,_] = env.env.env.env._robot.GetMotorVelocities()
    joint_pos[:,_] = env.env.env.env._robot.GetMotorAngles()

print('Possibly non-completed episode......')
print('episode reward:', episode_reward, "num_steps:", num_steps)
print(info)
print('=========================== Max joint angles')
print(np.amax(joint_pos, axis=1) )
print('=========================== Min joint angles')
print(np.amin(joint_pos, axis=1) )

t = np.linspace(0,10,len(act_vels))

# save velocities
np.savetxt('pybullet_vels.txt', np.vstack((t,act_vels)))

plt.plot(des_vels, label='des vel')
plt.plot(act_vels, label='act vel')
plt.legend()
plt.show()

# plt.plot(joint_taus.T)
# plt.legend(env.env.env.env._robot_config.JOINT_NAMES)
# plt.title('Torques')
# plt.show()

# plt.plot(joint_vels.T)
# plt.legend(env.env.env.env._robot_config.JOINT_NAMES)
# plt.title('velocities')
# plt.show()



ray.shutdown()