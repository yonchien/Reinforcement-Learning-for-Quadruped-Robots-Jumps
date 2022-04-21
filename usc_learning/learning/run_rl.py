import os
from datetime import datetime
import inspect 

import gym
import tensorflow as tf
import logging
# get rid of annoying tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# stable baselines
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1,PPO2, SAC, TD3
from stable_baselines.common.cmd_util import make_vec_env

from usc_learning.learning.algos.my_ppo2 import MYPPO2

# old robots if needed:
#from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
#from usc_learning.imitation_tasks.imitation_task import ImitationTaskAlienGoGymEnv
#from usc_learning.envs.LaikagoGymEnv import LaikagoGymEnv
#from usc_learning.envs.rex.walk_env import RexWalkEnv
#from usc_learning.envs.my_minitaur.my_minitaur_gym_env import MinitaurBulletEnv

# should really only use quadruped_master, supply configs here or manually
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
#from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
#from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
from usc_learning.envs.car.rcCarFlagRunGymEnv import RcCarFlagRunGymEnv
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config
from usc_learning.utils.learning_utils import CheckpointCallback

USE_ROS = False
USE_GAZEBO = False
USE_CAR = False

# 4 has been tested on mac and linux, right number of rollouts
NUM_ENVS = 1

LEARNING_ALG = 0
# -1 is MY_PPO2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC
# 3 is TD3

PARALLEL_FLAG = False
LOAD_PREV_MODEL = False

if USE_ROS:
	from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
	USE_GAS = True
	NUM_ENVS = 1
	PARALLEL_FLAG = False
	LEARNING_ALG = -1

if USE_GAZEBO:
	from usc_learning.ros_only.ros_gym import ROSGymEnv
	NUM_ENVS = 1
	PARALLEL_FLAG = False
	#LEARNING_ALG = 0


# Set all save locations
log_dir = "./logs/"
MODEL_SAVE_FILENAME = log_dir + 'ppo_aliengo'
ENV_STATS_FILENAME = os.path.join(log_dir, "vec_normalize.pkl")

#monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'
#save_path = './logs/intermediate_models/'+monitor_dir
save_path = 'logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(save_path, exist_ok=True)

# load info (example)
# load_dir =  "./logs/intermediate_models/" + "060220125255/" 
# load_stats_path = os.path.join(load_dir, "vec_normalize.pkl")
# load_model_name = os.path.join(load_dir, "rl_model_3000000_steps.zip")

# copy relevant files to save directory to provide snapshot of code changes
copy_files(save_path)
# checkpoint 
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path,name_prefix='rl_model', verbose=2)

# messy, to fix with argparse
if PARALLEL_FLAG:
	if USE_CAR:
		env = lambda: RcCarFlagRunGymEnv(render=False)
	else:
		env = lambda: QuadrupedGymEnv(render=False)
	env = make_vec_env(env, monitor_dir=save_path,n_envs=NUM_ENVS) #
	if LOAD_PREV_MODEL:
		env = VecNormalize.load(load_stats_path, env)
	else:
		env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

else:
	# setup things manually, for testing usually
	if USE_CAR:
		env = Monitor(RcCarFlagRunGymEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
	elif USE_ROS:
		env = Monitor(ROSQuadrupedGymEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
	elif USE_GAZEBO:
		env = Monitor(ROSGymEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
	else:
		env = Monitor(QuadrupedGymEnv(render=True, on_rack=True), filename=save_path+'monitor.csv', allow_early_resets=True)
	env = DummyVecEnv( [lambda: env])
	env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

# write env configs to reload (i.e if don't need to inspect whole code base)
write_env_config(save_path,env)

# Custom MLP policy of two layers of size _,_ each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200,100])
if USE_CAR:
	policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64,64])

if LEARNING_ALG <= 0:
	# PPO2 (Default)
	# ppo_config = {"gamma":0.99, "n_steps":512, "ent_coef":0.01, "learning_rate":2.5e-3, "vf_coef":0.5,
	# 				"max_grad_norm":0.5, "lam":0.95, "nminibatches":32, "noptepochs":4, "cliprange":0.2, "cliprange_vf":None,
	# 				"verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":policy_kwargs,
	# 				"full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	# lambda f: 2e-3 * f
	# these settings seem to be working well, with NUM_CPU=4
	n_steps = 1024
	nminibatches = int(n_steps / 64)
	learning_rate = lambda f: 1e-4 #5e-4 * f
	ppo_config = {"gamma":0.99, "n_steps":n_steps, "ent_coef":0.0, "learning_rate":learning_rate, "vf_coef":0.5,
					"max_grad_norm":0.5, "lam":0.95, "nminibatches":nminibatches, "noptepochs":10, "cliprange":0.2, "cliprange_vf":None,
					"verbose":1, "tensorboard_log": None,#"logs/ppo2/tensorboard/", 
					"_init_setup_model":True, "policy_kwargs":policy_kwargs,
					"full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	if LEARNING_ALG == -1:
		model = MYPPO2('MlpPolicy', env, **ppo_config)
	elif LOAD_PREV_MODEL:
		model = PPO2.load(load_model_name, env,**ppo_config)
	else:
		model = PPO2('MlpPolicy', env, **ppo_config)

elif LEARNING_ALG == 1:
	# PPO1
	# ppo_config = {"gamma":0.99, "timesteps_per_actorbatch":512, "clip_param":0.2, "entcoeff":0.01,
	# 				"optim_epochs":10, "optim_stepsize":1e-3, "optim_batchsize":64, "lam":0.95, "adam_epsilon":1e-5,
	# 				"schedule":'linear', "verbose":1, "tensorboard_log":"logs/ppo/tensorboard/", "_init_setup_model":True,
	# 				"policy_kwargs":policy_kwargs, "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	# ppo_config = {"gamma":0.99, "timesteps_per_actorbatch":512, "clip_param":0.2, "entcoeff":0.01,
	# 				"optim_epochs":10, "optim_stepsize":1e-4, "optim_batchsize":64, "lam":0.95, "adam_epsilon":1e-5,
	# 				"schedule":'constant', "verbose":1, "tensorboard_log":"logs/ppo/tensorboard/", "_init_setup_model":True,
	# 				"policy_kwargs":policy_kwargs, "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	ppo_config = {"gamma":0.99, "timesteps_per_actorbatch":4096, "clip_param":0.2, "entcoeff":0.0,
					"optim_epochs":10, "optim_stepsize":1e-4, "optim_batchsize":64, "lam":0.95, "adam_epsilon":1e-5,
					"schedule":'constant', "verbose":1, "tensorboard_log": None, #"logs/ppo/tensorboard/", 
					"_init_setup_model":True,
					"policy_kwargs":policy_kwargs, "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	model = PPO1('MlpPolicy', env, **ppo_config)

elif LEARNING_ALG == 2:
	# SAC
	# sac_config={"gamma"=0.99, "learning_rate"=3e-4, "buffer_size"=100000,
	# 			 "learning_starts"=100, "train_freq"=1, "batch_size"=64,
	# 			 "tau"=0.005, "ent_coef"='auto', "target_update_interval"=1,
	# 			 "gradient_steps"=1, "target_entropy"='auto', "action_noise"=None,
	# 			 "random_exploration"=0.0, "verbose"=0, "tensorboard_log"=None,
	# 			 "_init_setup_model"=True, "policy_kwargs"=None, "full_tensorboard_log"=False,
	# 			 "seed"=None, "n_cpu_tf_sess"=None}
	# check layers, from different format for SAC
	#policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[12])
	policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[200,100])
	sac_config={"gamma":0.99, "learning_rate":3e-4, "buffer_size":100000,
			 "learning_starts":1000, "train_freq":1, "batch_size":64,
			 "tau":0.005, "ent_coef":'auto', "target_update_interval":1,
			 "gradient_steps":1, "target_entropy":'auto', "action_noise":None,
			 "random_exploration":0.0, "verbose":1, "tensorboard_log":None,
			 "_init_setup_model":True, "policy_kwargs":policy_kwargs, "full_tensorboard_log":False,
			 "seed":None, "n_cpu_tf_sess":None}
	model = SAC('MlpPolicy', env, **sac_config)

elif LEARNING_ALG == 3:
	"""
	TODO: add TD3 ? 
	td3_config={"gamma":0.99, "learning_rate":0.0004, "buffer_size":50000,
	"learning_starts":1000, "train_freq":100, "batch_size":128,
	"tau":0.004,
	"gradient_steps":100, "action_noise":None,
	"random_exploration":0.0, "verbose":1, "tensorboard_log":"logs/td3/tensorboard/",
	"_init_setup_model":True, "policy_kwargs":policy_kwargs, "full_tensorboard_log":False,
	"seed":None, "n_cpu_tf_sess":None}
	"""
	policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[200,100])
	td3_config={"gamma":0.99, "learning_rate":3e-4, "buffer_size":100000,
			"learning_starts":1000, "train_freq":100, "batch_size":128,
			"tau":0.005,
			"gradient_steps":100, "action_noise":None,
			"random_exploration":0.0, "verbose":1, "tensorboard_log":None,
			"_init_setup_model":True, "policy_kwargs":policy_kwargs, "full_tensorboard_log":False,
			"seed":None, "n_cpu_tf_sess":None}
	model = TD3('MlpPolicy', env, **td3_config)
else:
	print('not a valid learning alg')
	sys.exit()

# Learn and save
model.learn(total_timesteps=100000000, log_interval=1,callback=checkpoint_callback)

# Don't forget to save the VecNormalize statistics when saving the agent
model.save(MODEL_SAVE_FILENAME)
env.save(ENV_STATS_FILENAME)

