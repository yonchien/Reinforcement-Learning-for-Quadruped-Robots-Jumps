import os
from datetime import datetime

import gym
import tensorflow as tf
import logging
# get rid of annoying tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1,PPO2, SAC
# from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.cmd_util import make_vec_env

from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
from usc_learning.imitation_tasks.imitation_task import ImitationTaskAlienGoGymEnv
from usc_learning.envs.LaikagoGymEnv import LaikagoGymEnv
from usc_learning.envs.rex.walk_env import RexWalkEnv
from usc_learning.envs.my_minitaur.my_minitaur_gym_env import MinitaurBulletEnv
# from usc_learning.envs.my_minitaur.quadruped_gym_env import QuadrupedGymEnv
# from usc_learning.envs.my_minitaur.quadruped_gym_env_v2 import QuadrupedGymEnvV2
#from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
from quadruped_gym_env import QuadrupedGymEnv
#from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
from usc_learning.envs.car.rcCarFlagRunGymEnv import RcCarFlagRunGymEnv

import pybullet_envs
import pybullet_data
"""
def pybullet_minitaur():
  randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
  env = functools.partial(minitaur_gym_env.MinitaurBulletEnv,
                          accurate_motor_model_enabled=True,
                          motor_overheat_protection=True,
                          pd_control_enabled=True,
                          env_randomizer=randomizer,
                          render=False)
  max_length = 1000
  steps = 3e7  # 30M
  return locals()
"""
NUM_ENVS = 4 # 4 has been tested on mac and linux, right numbner of rollouts

LEARNING_ALG = 0
# 0 is PPO2
# 1 is PPO1
# 2 is SAC

ROBOT = 7
# 0 is aliengo
# 1 is laikago
# 2 is minitaur (either default or personal)
# 3 is rex WALK (spot micro)
# 4 is rex STANDUP
# 5 is Quadruped (general) [laikago]
# 6 is Quadruped-v2 [laikago]
# 7 is RcCarFlagRunGymEnv

PARALLEL_FLAG = True
IMITATION_FLAG = False
LOAD_PREV_MODEL = False

# Set all save locations
log_dir = "./logs/"
MODEL_SAVE_FILENAME = log_dir + 'ppo_aliengo'
ENV_STATS_FILENAME = os.path.join(log_dir, "vec_normalize.pkl")

monitor_dir = datetime.now().strftime("%m%d%y%H%M%S") + '/'
save_path = './logs/intermediate_models/'+monitor_dir
os.makedirs(save_path, exist_ok=True)

# load info
load_dir =  "./logs/intermediate_models/" + "060220125255/" 
load_stats_path = os.path.join(load_dir, "vec_normalize.pkl")
load_model_name = os.path.join(load_dir, "rl_model_3000000_steps.zip")


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
		return True


checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_path,name_prefix='rl_model', verbose=2)

# messy, to fix with argparse
if PARALLEL_FLAG:
	if IMITATION_FLAG:
		#env = ImitationTaskAlienGoGymEnv(isrender=False)
		env = lambda: ImitationGymEnv(render=False)
	else:
		if ROBOT == 0: 
			env = lambda: AlienGoGymEnv(isrender=False)
		elif ROBOT == 1:
			env = lambda: LaikagoGymEnv()
		elif ROBOT == 2:
			#env = "MinitaurBulletEnv-v0"
			env = lambda: MinitaurBulletEnv()
		elif ROBOT == 3:
			env = lambda: RexWalkEnv()
		elif ROBOT == 4:
			raise ValueError("not implemented robot 4")
		elif ROBOT == 5:
			env = lambda: QuadrupedGymEnv(render=False)
		elif ROBOT == 7:
			env = lambda: RcCarFlagRunGymEnv()
		else:
			env = lambda: QuadrupedGymEnvV2()
		

	env = make_vec_env(env, monitor_dir=save_path,n_envs=NUM_ENVS) #
	if LOAD_PREV_MODEL:
		env = VecNormalize.load(load_stats_path, env)
	else:
		env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
else:
	# setup things manually
	if IMITATION_FLAG:
		#env = Monitor(ImitationTaskAlienGoGymEnv(isrender=False), filename=save_path+'monitor.csv', allow_early_resets=True)
		env = Monitor(ImitationGymEnv(render=False), filename=save_path+'monitor.csv', allow_early_resets=True)
	else:
		if ROBOT == 0:
			env = Monitor(AlienGoGymEnv(isrender=False), filename=save_path+'monitor.csv', allow_early_resets=True)
		elif ROBOT == 1:
			env = Monitor(LaikagoGymEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
		elif ROBOT == 2:
			env = Monitor("MinitaurBulletEnv-v0", filename=save_path+'monitor.csv', allow_early_resets=True)
		elif ROBOT == 5:
			env = Monitor(QuadrupedGymEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
		else:
			env = Monitor(RexWalkEnv(), filename=save_path+'monitor.csv', allow_early_resets=True)
	env = DummyVecEnv( [lambda: env])
	env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Custom MLP policy of two layers of size _,_ each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200,100])

if LEARNING_ALG == 0:
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
					"verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":policy_kwargs,
					"full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
	if LOAD_PREV_MODEL:
		model = PPO2.load(load_model_name, env,**ppo_config)
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
					"schedule":'constant', "verbose":1, "tensorboard_log":"logs/ppo/tensorboard/", "_init_setup_model":True,
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
	policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[12])
	sac_config={"gamma":0.99, "learning_rate":3e-4, "buffer_size":10000,
			 "learning_starts":1000, "train_freq":1, "batch_size":64,
			 "tau":0.005, "ent_coef":'auto', "target_update_interval":1,
			 "gradient_steps":1, "target_entropy":'auto', "action_noise":None,
			 "random_exploration":0.0, "verbose":1, "tensorboard_log":None,
			 "_init_setup_model":True, "policy_kwargs":policy_kwargs, "full_tensorboard_log":False,
			 "seed":None, "n_cpu_tf_sess":None}
	model = SAC('MlpPolicy', env, **sac_config)

else:
	print('not a valid learning alg')
	sys.exit()

model.learn(total_timesteps=100000000, log_interval=1,callback=checkpoint_callback)

# Don't forget to save the VecNormalize statistics when saving the agent
model.save(MODEL_SAVE_FILENAME)
env.save(ENV_STATS_FILENAME)

