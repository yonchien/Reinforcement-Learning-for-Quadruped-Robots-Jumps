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
from stable_baselines import PPO1,PPO2, SAC
from stable_baselines.common.cmd_util import make_vec_env

from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv

# envs
import usc_learning.envs.quadruped_master as quadruped_master
import usc_learning.imitation_tasks as imitation_tasks
# utils
from usc_learning.utils.file_utils import copy_files, write_env_config
from usc_learning.utils.learning_utils import CheckpointCallback

""" Imitation only. """
# 4 has been tested on mac and linux, right numbner of rollouts
NUM_ENVS = 1

LEARNING_ALG = 2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC

if LEARNING_ALG > 0:
	save_freq = 1000
else:
	save_freq = 10000

USE_LSTM = False


PARALLEL_FLAG = False
LOAD_PREV_MODEL = False

# Set all save locations
log_dir = "./logs/"
MODEL_SAVE_FILENAME = log_dir + 'ppo_a1'
ENV_STATS_FILENAME = os.path.join(log_dir, "vec_normalize.pkl")

monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'
save_path = 'logs/intermediate_models/'+monitor_dir
os.makedirs(save_path, exist_ok=True)

if LOAD_PREV_MODEL:
	# load info
	load_dir = './logs/intermediate_models/102420223827/'
	# load_dir =  "./logs/intermediate_models/" + "060220125255/" 
	load_stats_path = os.path.join(load_dir, "vec_normalize.pkl")
	load_model_name = os.path.join(load_dir, "rl_model_49000_steps.zip")

# copy relevant files to save directory to provide snapshot of code changes
copy_files(save_path)
# checkpoint

checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path,name_prefix='rl_model', verbose=2)

# messy, to fix with argparse
if PARALLEL_FLAG:
	env = lambda: ImitationGymEnv(render=False)
	env = make_vec_env(env, monitor_dir=save_path,n_envs=NUM_ENVS) #
	if LOAD_PREV_MODEL:
		env = VecNormalize.load(load_stats_path, env)
	else:
		env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

else:
	# setup things manually (for debugging usually)
	env = Monitor(ImitationGymEnv(render=False), filename=save_path+'monitor.csv', allow_early_resets=True)
	env = DummyVecEnv( [lambda: env])
	if LOAD_PREV_MODEL:
		env = VecNormalize.load(load_stats_path, env)
	else:
		# confirmed: DON'T normalize reward
		env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.) 
	#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)

# write env configs to reload (i.e if don't need to inspect whole code base)
write_env_config(save_path,env)

# Custom MLP policy of two layers of size _,_ each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512,512])
lstm_policy_kwargs = dict() #dict(act_fun=tf.nn.tanh, net_arch=[200,200])

if LEARNING_ALG == 0:
	# PPO2 (Default)
	# ppo_config = {"gamma":0.99, "n_steps":512, "ent_coef":0.01, "learning_rate":2.5e-3, "vf_coef":0.5,
	# 				"max_grad_norm":0.5, "lam":0.95, "nminibatches":32, "noptepochs":4, "cliprange":0.2, "cliprange_vf":None,
	# 				"verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":policy_kwargs,
	# 				"full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	# lambda f: 2e-3 * f
	# these settings seem to be working well, with NUM_CPU=4
	n_steps = 1024
	nminibatches = int(n_steps / 64)
	# n_steps = 100
	# nminibatches = int(n_steps / 2)
	learning_rate = lambda f: 1e-4 #5e-4 * f
	ppo_config = {"gamma":0.99, "n_steps":n_steps, "ent_coef":0.0, "learning_rate":learning_rate, "vf_coef":0.5,
					"max_grad_norm":0.5, "lam":0.95, "nminibatches":nminibatches, "noptepochs":10, "cliprange":0.2, "cliprange_vf":None,
					"verbose":1, "tensorboard_log": None, #"logs/ppo2/tensorboard/", 
					"_init_setup_model":True, "policy_kwargs":policy_kwargs,
					"full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	if LOAD_PREV_MODEL:
		model = PPO2.load(load_model_name, env,**ppo_config)
	elif USE_LSTM:
		ppo_config["policy_kwargs"] = lstm_policy_kwargs
		ppo_config["nminibatches"] = NUM_ENVS
		model = PPO2('MlpLstmPolicy', env, **ppo_config)
	else:
		model = PPO2('MlpPolicy', env, **ppo_config)
elif LEARNING_ALG == 1:
	# PPO1
	# ppo_config = {"gamma":0.99, "timesteps_per_actorbatch":512, "clip_param":0.2, "entcoeff":0.01,
	# 				"optim_epochs":10, "optim_stepsize":1e-3, "optim_batchsize":64, "lam":0.95, "adam_epsilon":1e-5,
	# 				"schedule":'linear', "verbose":1, "tensorboard_log":"logs/ppo/tensorboard/", "_init_setup_model":True,
	# 				"policy_kwargs":policy_kwargs, "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	ppo_config = {"gamma":0.99, "timesteps_per_actorbatch":512, "clip_param":0.2, "entcoeff":0.01,
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
	# policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[200,100])
	policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[512,512])
	sac_config={"gamma":0.99, "learning_rate":3e-4, "buffer_size":100000,
			 "learning_starts":100, "train_freq":1, "batch_size":64,
			 "tau":0.005, "ent_coef":'auto', "target_update_interval":1,
			 "gradient_steps":1, "target_entropy":'auto', "action_noise":None,
			 "random_exploration":0.0, "verbose":1, "tensorboard_log":None,
			 "_init_setup_model":True, "policy_kwargs":policy_kwargs, "full_tensorboard_log":False,
			 "seed":None, "n_cpu_tf_sess":None}

	if LOAD_PREV_MODEL:
		model = SAC.load(load_model_name, env,**sac_config)
	else:
		model = SAC('MlpPolicy', env, **sac_config)

else:
	print('not a valid learning alg')
	sys.exit()

model.learn(total_timesteps=100000000, log_interval=1,callback=checkpoint_callback)

# Don't forget to save the VecNormalize statistics when saving the agent
model.save(MODEL_SAVE_FILENAME)
env.save(ENV_STATS_FILENAME)

