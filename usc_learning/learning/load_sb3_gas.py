import os, sys
import gym
import numpy as np
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import tensorflow
import logging
import inspect 
# get rid of annoying tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# stable baselines
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# should really only use quadruped_master, supply configs here or manually
#from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config, get_latest_model
#from usc_learning.utils.learning_utils import CheckpointCallback
#from stable_baselines3.bench.monitor import load_results

# utils
from usc_learning.utils.utils import nicePrint, nicePrint2D
from usc_learning.utils.utils import plot_results
from usc_learning.utils.file_utils import get_latest_model, read_env_config, load_all_results



USE_ROS = False

LEARNING_ALG = 2
# -1 is MYPPO2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC
# 3 is TD3

interm_dir = "./logs/intermediate_models/"

config_dir = interm_dir + '111120225601'
log_dir = interm_dir + '111120225601/9/111120225601'

config_dir = interm_dir + '111220022937'
log_dir = interm_dir + '111220022937/9/111220022937'

# PPO
# config_dir = interm_dir + '111220003950'
# log_dir = interm_dir + '111220003950/9/111220003950'

# check thes
# PPO
# 111220022948
# SAC
# 111220022937



config_dir = '111220133018'
config_dir = '111220154130'

config_dir = '111220172751'
# config_dir = '111220171937'

# Td3
# config_dir = '111220192547'
config_dir = '111220203607'
config_dir = '111220221544'

config_dir = '111320002744' # this actruall GAS

# config_dir = '111220203315'
config_dir = '111220234012'
# config_dir = '111220230448'

config_dir = '111220234111'
log_dir = '111220234111/7/111320030236'

config_dir = '111220234012'
log_dir = '111220234012/9/111220234012'

config_dir = '111220230448'
log_dir = '111220230448/9/111220230448'

# pretty good SAC
config_dir = '111320031122'
log_dir = '111320031122/4/111320031122'

# td3, weird
config_dir = '111320031128'
log_dir = '111320031128/4/111320031128'

config_dir = '111320111033'
log_dir = '111320111033/4/111320111033'

# test GAS
# config_dir = '111320133457'
# log_dir = '111320133457/4/111320152157'

# config_dir = '111320110955'
# log_dir = '111320110955/4/111320110955'

# config_dir = '111320133457'
# #log_dir = '111320133457/2/111320144533'

# log_dir = '111320133457/3/111320150353'

config_dir = '111420011645'
log_dir = '111420011645/4/111420041217'

# td3, good, but on knees
config_dir = '111420141935'
log_dir = '111420141935/4/111420181249'

#sac
config_dir = '111420141923'
log_dir = '111420141923/4/111420154557'

config_dir = '111520214525'
log_dir = '111520214525/4/111520224233'

config_dir = '111520214525'
log_dir = '111520214525/4/111620002603'


# these 3 make progress, but the results are not very believeable  

# poor, not great curve
config_dir = '111620003917'
log_dir = '111620003917/4/111620021828'

# good curves, no randomization 
config_dir = '111520214525'
log_dir = '111520214525/4/111620130803'

# on rear legs only 
config_dir = '111620013431'
log_dir = '111620013431/4/111620050459'

config_dir = '111620142912'
log_dir= '111620142912/4/111620161931'

# really good! energy weight .05, did fall at beginning, w dynamics randomizaiton 
config_dir = '111620162558'
log_dir = '111620162558/4/111620204747'

# energy weigth .01, not as believable as above 
config_dir = '111620142912'
log_dir = '111620142912/4/111620212542'

# energy weight .1, not bad
config_dir = '111620142906'
log_dir = '111620142906/3/111620212340'

# running but not very believable
config_dir = '111720015745'
log_dir = '111720015745/4/111720033839'

# .005 weight energy, good
config_dir = '111720054150'
log_dir= '111720054150/4/111720111430'

# 0 weight energy, too jittery
# config_dir = '111720054226'
# log_dir = '111720054226/4/111720074024'

config_dir = '112120135500'
log_dir = '112120135500/1/112120151444'

config_dir = '112720174430'
log_dir = '112720174430/4/112720174430'

# TD3
# config_dir = '112720184928'
# log_dir = '112720184928/4/112720184928'

config_dir = '112720235848'
log_dir = '112720235848/4/112720235848'

# w noise, XY only
config_dir = '112820031332'
log_dir = '112820031332/4/112820031332'

config_dir = '112720235848'
log_dir = '112720235848/4/112720235848'

config_dir = '112820162642'
log_dir = '112820162642/4/112820162642'

config_dir = '112920003251'
log_dir = '112920003251/4/112920003251'

config_dir = '112920003134'
log_dir = '112920003134/4/112920003134'

config_dir = '112920042152'
log_dir = '112920042152/4/112920042152'

config_dir = '112920042202'
log_dir = '112920042202/4/112920042202'

config_dir = '112920150345'
log_dir = '112920150345/4/112920150345'

config_dir = '112920182509'
log_dir = '112920182509/4/112920182509'

config_dir = '113020032823'
log_dir = '113020032823/4/113020032823'

# config_dir = '113020033747'
# log_dir = '113020033747/4/113020033747'

config_dir = '113020180035'
log_dir = '113020180035/4/113020180035'

# config_dir = '113020181311'
# log_dir = '113020181311/4/113020181311'

config_dir = '120120011355'
log_dir = '120120011355/4/120120011355'

config_dir = '120120011405'
log_dir = '120120011405/4/120120011405'

# this actually works well
config_dir = '120220011534'
log_dir= '120220011534/4/120220011534'

# config_dir = '120220012510'
# log_dir = '120220012510/4/120220012510'

log_dir = '120220232958/4/120220232958'
config_dir = '120220232958'

log_dir = '120220233018/4/120220233018'
config_dir = '120220233018'

log_dir = '120420001859/4/120420001859'
config_dir = '120420001859'

# log_dir = '120420001921/4/120420001921'
# config_dir = '120420001921'

log_dir = '120420122917/4/120420122917'
config_dir = '120420122917'

config_dir = '120420122931'
config_dir = '120520014225' # xy
# config_dir = '120520014149'

config_dir = '121420223133' # PMTG_XYZ_ONLY_JOINT
# config_dir = '121420223255' # PMTG_FULL_JOINT
config_dir = '121520140541' # w cost function penalizing orientation

########################################################## jan 2021
config_dir = '012121094540'
# config_dir = '012121151113'
config_dir = '012121222613'
# config_dir = '012221160404'
config_dir = '012221181448'
# config_dir = '012221233010'
config_dir = '012421003405'
# config_dir = '012421114529' # good
config_dir = '012421174416'
# reboot
config_dir = '012421233732'
# config_dir = '012421233708'
config_dir = '012521104606'
# config_dir = '012521133651'
config_dir = '021421031913'
config_dir = '021421035001'

log_dir = config_dir + '/4/' + config_dir
#log_dir = config_dir + '/7/' + config_dir
###########################################################################################
config_dir = interm_dir + config_dir
log_dir = interm_dir + log_dir

# get env config if available
try:
	env_config = read_env_config(log_dir)
	# sys.path.append(log_dir)
	sys.path.insert(0,config_dir)
	import configs_a1 as robot_config
	from quadruped_gym_env import QuadrupedGymEnv
	env_config['robot_config'] = robot_config
except:
	env_config = {}
	raise ValueError('Could not load env config and/or gym')

env_config['render'] = True
env_config['record_video'] = False
# env_config['grow_action_space'] = True
env_config['randomize_dynamics'] = False

for k,v in env_config.items():
	print(k, v)

################################
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', 'AlienGo PPO')
plt.show() #block=False

load_all_results(config_dir)
plt.show()

# reconstruct env 
if USE_ROS:
	env = lambda: ROSQuadrupedGymEnv(render=True)
else:
	env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
# do not update stats at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False


if LEARNING_ALG == 0:
	model = PPO.load(model_name, env)
elif LEARNING_ALG == 2:
	model = SAC.load(model_name, env)
elif LEARNING_ALG == 3:
	model = TD3.load(model_name, env)
print("\n Loaded model", model_name, "\n")

outside_torque_limits_counter = 0
obs = env.reset()
episode_reward = 0
for i in range(2000):
	#print('TIME',env.envs[0].env.get_sim_time())
	action, _states = model.predict(obs)
	#print(action)
	# print(obs)

	#print("ACTUAL ANGLES", env.envs[0].env._robot.GetJointAngles())
	xyz, Fn = env.envs[0].env._robot.GetXYZFn()

	x = xyz[0::3]
	y = xyz[1::3]
	z = xyz[2::3]
	#print('x',xyz)
	#print('Fn', Fn)
	real_xyz = env.envs[0].env._robot.save_real_xyz
	real_x = real_xyz[0::3]
	real_y = real_xyz[1::3]
	real_z = real_xyz[2::3]
	real_Fn = -np.array(env.envs[0].env._robot.save_real_Fn)

	obs, rewards, dones, info = env.step(action)
	episode_reward += rewards
	if dones:
		#break
		#obs = env.reset()
		print('*'*80)
		print('episode_reward', episode_reward)
		print('*'*80)
		episode_reward = 0

	#if ROBOT > 1: # minitaur or rex
	#env.render()
	time.sleep(0.01)

#xyz = env.envs[0].env._robot.save_xyz
#Fn = env.envs[0].env._robot.save_Fn
# xyz, Fn =  env.envs[0].env._robot.GetXYZFn()

# x = xyz[0::3]
# y = xyz[1::3]
# z = xyz[2::3]
# print('x',xyz)
# print('Fn', Fn)
t = np.linspace(0,5,len(x))

plt.plot(t,x, label='x')
plt.plot(t,y, label='y')
plt.plot(t,z, label='z')
plt.plot(t,real_x, label='real_x')
plt.plot(t,real_y, label='real_y')
plt.plot(t,real_z, label='real_z')
plt.legend()
plt.show()

plt.plot(t,Fn[:len(x)], label='Fn')
plt.plot(t,real_Fn[:len(x)], label='real_Fn',linestyle='--')
plt.legend()
plt.show()