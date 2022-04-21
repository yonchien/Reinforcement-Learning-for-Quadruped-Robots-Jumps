import os, sys
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

#from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1,PPO2, SAC, TD3
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.bench.monitor import load_results
#from stable_baselines.results_plotter import plot_results

# old robots if needed:
#from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
#from usc_learning.imitation_tasks.imitation_task import ImitationTaskAlienGoGymEnv
#from usc_learning.envs.LaikagoGymEnv import LaikagoGymEnv
#from usc_learning.envs.rex.walk_env import RexWalkEnv
#from usc_learning.envs.my_minitaur.my_minitaur_gym_env import MinitaurBulletEnv

from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv

#from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv

# utils
from usc_learning.utils.utils import nicePrint, nicePrint2D
from usc_learning.utils.utils import plot_results
from usc_learning.utils.file_utils import get_latest_model, read_env_config

from usc_learning.learning.algos.my_ppo2 import MYPPO2


USE_ROS = False

LEARNING_ALG = 2
# -1 is MYPPO2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC
# 3 is TD3

if USE_ROS:
	from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv 
	LEARNING_ALG = -1

interm_dir = "./logs/intermediate_models/"
log_dir = "./logs/intermediate_models/" + "060620191048/" # good minitaur without early stopping
log_dir = "./logs/intermediate_models/" + "060820124702" # good laikago, works
log_dir = "./logs/intermediate_models/" + "061020105659" # good laikago (quadruped v2), works
log_dir = "./logs/intermediate_models/" + "062320001232" # current aliengo impedance control (WORKS GREAT)


log_dir = interm_dir + '070120151616/' # IK
log_dir = interm_dir + '070120214821/' # good Impedance
log_dir = interm_dir + '070220215756'

log_dir = interm_dir + '070520230703' # impedance_gains
log_dir = interm_dir + '070620103721' # impedance

log_dir = interm_dir + '070720190536' # a1
log_dir = interm_dir + '070820111437' # new aliengo

log_dir = interm_dir + '070820213616' # aliengo no terrain
log_dir = interm_dir + '070920123804' # ''

log_dir = interm_dir + '071020014919/' # A1

log_dir = interm_dir + '071720181220'

# stand up
log_dir = interm_dir + '072120143907'
#log_dir = interm_dir + '072120170645'
log_dir = interm_dir + '072120173035'
log_dir = interm_dir + '072120195334'

# ROS
log_dir = interm_dir + '072420134000'

# SAC
#log_dir = interm_dir + '072320221219'
log_dir = interm_dir + '081420150008'
log_dir = interm_dir + '081420171651'

log_dir = interm_dir + '081420150008'

# ROSGym
log_dir = interm_dir + '101720144218' # didnt learn
# log_dir = interm_dir + '101920102240'
# log_dir = interm_dir + '101920150933' # learning
log_dir = interm_dir + '101920204446' # learning something

log_dir = interm_dir + '102120181500'

log_dir = interm_dir + '102120195815'
log_dir = interm_dir + '102220033407'
# learn velocity only
log_dir = interm_dir + '102220145743'

# get env config if available
try:
	env_config = read_env_config(log_dir)
	sys.path.append(log_dir)

	# check env_configs.txt file to see whether to load aliengo or a1
	with open(log_dir + '/env_configs.txt') as configs_txt:
		if 'aliengo' in configs_txt.read():
			import configs_aliengo as robot_config
		else:
			import configs_a1 as robot_config

	env_config['robot_config'] = robot_config
except:
	env_config = {}
env_config['render'] = True
env_config['record_video'] = False

stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', 'AlienGo PPO')
plt.show() #block=False

sys.exit()
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

if LEARNING_ALG == -1:
	model = MYPPO2.load(model_name, env)
elif LEARNING_ALG == 0:
	model = PPO2.load(model_name, env)
elif LEARNING_ALG == 1:
	model = PPO1.load(model_name, env)
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
	#action = obs
	#print('*'*80,'\n Observation:')
	# nicePrint2D(obs)
	# print('Action:')
	# nicePrint2D(action)
	# print('-'*80,)
	# print('ACTION PROBABILITY mean:') 
	# nicePrint2D(model.action_probability(obs)[0])
	# print('ACTION PROBABILITY STD:') 
	# nicePrint2D(model.action_probability(obs)[1])
	# print('-'*80,)
	# print('base',np.asarray(env.envs[0].env._robot.GetBasePosition()))
	# print('RPY',env.envs[0].env._robot.GetBaseOrientationRollPitchYaw())
	# print('Angles',env.envs[0].env._robot.GetMotorAngles() )
	#print('Torques',env.envs[0].env._robot.GetMotorTorques() )
	actual_torque = env.envs[0].env._robot.GetMotorTorques()
	torque_ranges = np.asarray( [44,44,55] * 4 )
	if any( abs(actual_torque) > torque_ranges ):
		outside_torque_limits_counter += 1
	#nicePrint(env.envs[0].env._robot.GetMotorTorques())
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

	#if ROBOT > 1: # minitaur or rex
	#env.render()
	time.sleep(0.01)

print('num instances torque too big: ', outside_torque_limits_counter)
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