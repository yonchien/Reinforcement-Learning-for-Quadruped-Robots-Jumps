import os
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import tensorflow
import logging
# get rid of annoying tensorflow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1,PPO2, SAC
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.bench.monitor import load_results
#from stable_baselines.results_plotter import plot_results

from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
from usc_learning.imitation_tasks.imitation_task import ImitationTaskAlienGoGymEnv
from usc_learning.envs.LaikagoGymEnv import LaikagoGymEnv
from usc_learning.utils.utils import nicePrint, nicePrint2D
from usc_learning.utils.utils import plot_results

#import pybullet_envs
#import pybullet_data
# from pybullet_envs.minitaur.envs.minitaur_gym_env import MinitaurGymEnv
#from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from usc_learning.envs.my_minitaur.my_minitaur_gym_env import MinitaurBulletEnv
from usc_learning.envs.rex.walk_env import RexWalkEnv
#from usc_learning.envs.my_minitaur.quadruped_gym_env import QuadrupedGymEnv
#rom usc_learning.envs.my_minitaur.quadruped_gym_env_v2 import QuadrupedGymEnvV2
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
from usc_learning.envs.car.rcCarFlagRunGymEnv import RcCarFlagRunGymEnv

LEARNING_ALG = 0
# 0 is PPO2
# 1 is PPO1
# 2 is SAC

ROBOT = 5
# 0 is aliengo
# 1 is laikago
# 2 is minitaur (either default or personal)
# 3 is rex WALK (spot micro)
# 4 is rex STANDUP (not implemented yet)
# 5 is Quadruped (general)
# 6 is Quadruped-v2 [laikago]
# 7 is RcCarFlagRunGymEnv

IMITATION_FLAG = False
LAIKAGO_FLAG = False
MINITAUR_FLAG = False


def get_latest_model(path):
	""" Returns most recent model saved in path directory. """
	files = os.listdir(path)
	paths = [os.path.join(path, basename) for basename in files if basename.endswith('.zip')]
	return max(paths, key=os.path.getctime)

log_dir = "./logs/intermediate_models/" + "060620191048/" # good minitaur without early stopping
log_dir = "./logs/intermediate_models/" + "060820124702" # good laikago, works
log_dir = "./logs/intermediate_models/" + "061020105659" # good laikago (quadruped v2), works

log_dir = "./logs/intermediate_models/" + "062120125305/"

log_dir = "./logs/intermediate_models/" + "062820102932/"

#log_dir = "./logs/intermediate_models/" + "062920011506" # rc car



stats_path = os.path.join(log_dir, "vec_normalize.pkl")
#model_name = log_dir + 'rl_model_210000_steps.zip'
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', 'AlienGo PPO')
plt.show() #block=False

if IMITATION_FLAG:
	#env = lambda: ImitationTaskAlienGoGymEnv(isrender=False)
	env = lambda: ImitationGymEnv(render=True)
else:
	if ROBOT == 0:
		env = lambda: AlienGoGymEnv(isrender=True)
	elif ROBOT == 1:
		env = lambda: LaikagoGymEnv(isrender=True)
	elif ROBOT == 2:
		env = lambda: MinitaurBulletEnv(render=True) #"MinitaurBulletEnv-v0"
	elif ROBOT == 3:
			env = lambda: RexWalkEnv(render=True)
	elif ROBOT == 4:
			raise ValueError("not implemented robot 4")
	elif ROBOT == 5:
		env = lambda: QuadrupedGymEnv(render=True)
	elif ROBOT == 7:
		env = lambda: RcCarFlagRunGymEnv(render=True)
	else:
		env = lambda: QuadrupedGymEnvV2(render=True)

# if MINITAUR_FLAG:
# 	env = make_vec_env(env, n_envs=1,env_kwargs={'render':True}) 
# else:
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
# do not update stats at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

if LEARNING_ALG == 0:
	model = PPO2.load(model_name, env)
elif LEARNING_ALG == 1:
	model = PPO1.load(model_name, env)
elif LEARNING_ALG == 2:
	model = SAC.load(model_name, env)
print("\n Loaded model", model_name, "\n")

outside_torque_limits_counter = 0
obs = env.reset()
for i in range(1000):
	action, _states = model.predict(obs)
	if ROBOT < 7:
		print('TIME',env.envs[0].env.get_sim_time())
		
		#action = obs
		print('*'*80,'\n Observation:')
		nicePrint2D(obs)
		print('Action:')
		nicePrint2D(action)
		print('-'*80,)
		print('ACTION PROBABILITY mean:') 
		nicePrint2D(model.action_probability(obs)[0])
		print('ACTION PROBABILITY STD:') 
		nicePrint2D(model.action_probability(obs)[1])
		print('-'*80,)
		print('base',np.asarray(env.envs[0].env._robot.GetBasePosition()))
		print('RPY',env.envs[0].env._robot.GetBaseOrientationRollPitchYaw())
		print('Angles',env.envs[0].env._robot.GetMotorAngles() )
		print('Torques' )
		torque_ranges = np.asarray( [44,44,55] * 4 )
		actual_torque = env.envs[0].env._robot.GetMotorTorques()
		if any( abs(actual_torque) > torque_ranges ):
			outside_torque_limits_counter += 1
		nicePrint(env.envs[0].env._robot.GetMotorTorques())
	#nicePrint(env.envs[0].env._robot.GetBaseVelocity())

	#action = [env.envs[0].env._robot.GetDefaultInitJointPose()]
	#nicePrint(env.envs[0].env._robot.GetTrueMotorAngles())
	# laikago working actions
	# if 2D env
	#	action = np.array([[0.67,-1.25]*4])
	# else
	#	action = np.array([[0, 0.67,-1.25]*4])

	des_pos = np.array([  0.0, 0.65, -1.2 ] * 4)
	#des_pos = np.zeros(12)
	des_pos = np.array([0.0, 1.2217304764, -2.77507351067 ] * 4)
	#action = [env.envs[0].env._robot._UNscalePositionsToActions( des_pos )]
	#print("ACTUAL ANGLES", env.envs[0].env._robot.GetJointAngles())

	obs, rewards, dones, info = env.step(action)
	if dones:
		#obs = env.reset()
		print('*'*80, '\nreset\n','*'*80)

	if ROBOT > 1: # minitaur or rex
		env.render()
	time.sleep(0.01)

print('num instances torque too big: ', outside_torque_limits_counter)
