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
from usc_learning.utils.utils import nicePrint, nicePrint2D
from usc_learning.utils.utils import plot_results

from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv
import json
import sys
from usc_learning.utils.file_utils import get_latest_model, read_env_config

def plot_all_log_dirs_after(path, filename):
	""" Returns most recent model saved in path directory. """
	files = os.listdir(path)
	# return max(paths, key=os.path.getctime)
	# paths = [os.path.join(path, basename) for basename in files if int(basename) >= int(basename)]
	paths = [os.path.join(path, basename) for basename in files if int(basename) >= int(filename)]
	for file in paths:
		print(file)
		plot_results([file] , 10e10, 'timesteps', file)
		plt.show() #block=False


"""Load imitation. """

LEARNING_ALG = 2
# 0 is PPO2
# 1 is PPO1
# 2 is SAC


# paper
log_dir = '103020152802' 

#######################################################################################
if 'logs' not in log_dir:
	log_dir = interm_dir + log_dir

# get env config if available
try:
	env_config = read_env_config(log_dir)
	sys.path.append(log_dir)
	# sys.path.insert(0,log_dir)
	import configs_a1 as robot_config
	from imitation_gym_env import ImitationGymEnv
	env_config['robot_config'] = robot_config
except:
	env_config = {}
	raise ValueError('Could not load env config and/or gym')
#env_config = {}
env_config['land_and_settle'] = True
env_config['render'] = True
# env_config['opt_trajs_path'] = 'test_data'
# env_config['record_video'] = True
# env_config['task_mode'] ="DEFAULT_CARTESIAN"
# env_config['enable_action_filter'] = True
# env_config['randomize_dynamics'] = False
# env_config["set_kp_gains"] = "LOW"

env_config['test_heights'] = np.array([ [0.01, 0.05, 0.01, 0.1] ,
                                   [0.05, 0.01, 0.1 , 0.01]]).T


stats_path = os.path.join(log_dir, "vec_normalize.pkl")
#model_name = log_dir + 'rl_model_210000_steps.zip'
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', 'AlienGo PPO')
plt.show() #block=False

#sys.exit()

# set up env
env = lambda: ImitationGymEnv(**env_config)
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


# import pdb
# pdb.set_trace()
obs = env.reset()
action, _states = model.predict(obs)
#sys.exit()

print(env.envs[0].env)
print(env.envs[0])
print(env)
print('mean', env.obs_rms.mean)
print('var ', env.obs_rms.var)


# write_out_path = '/home/guillaume/catkin_ws/src/USC-AlienGo-Software/laikago_ros/jumping/'
# env.envs[0].env._traj_task.writeoutTraj(write_out_path)

#sys.exit()


""" 
SAC
input is:
> /home/guillaume/Documents/Research/DRL/venv/lib/python3.6/site-packages/stable_baselines/sac/policies.py(265)step()
-> return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
(Pdb) self.obs_ph
<tf.Tensor 'input/input/Ob:0' shape=(?, 125) dtype=float32>
(Pdb) 


Output:
(Pdb) self.deterministic_policy
<tf.Tensor 'model/Tanh:0' shape=(?, 96) dtype=float32>
so, should be
model.policy_tf.deterministic_policy


PPO2
Input
> /home/guillaume/Documents/Research/DRL/venv/lib/python3.6/site-packages/stable_baselines/common/policies.py(576)step()
-> {self.obs_ph: obs})
(Pdb) self.obs_ph
<tf.Tensor 'input/Ob:0' shape=(?, 341) dtype=float32>

Output
> /home/guillaume/Documents/Research/DRL/venv/lib/python3.6/site-packages/stable_baselines/common/policies.py(575)step()
-> action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
(Pdb) self.action
<tf.Tensor 'output/add:0' shape=(?, 12) dtype=float32>



"""

# write out in general?
#write_out_path = './exported_model/'

WRITE_OUT_GRAPH = False
if WRITE_OUT_GRAPH:
	import tensorflow as tf
	import shutil
	# i = tf.initializers.global_variables()
	# print('variables', tf.global_variables())
	#for op in 
	print(tf.trainable_variables())
	with model.graph.as_default():
		# i = tf.initializers.global_variables()
		# print('variables:')
		# for v in  tf.global_variables():
		# 	print(v)
		print(tf.trainable_variables())

		checkpoint_name = os.path.join(write_out_path, 'my_model')
		if os.path.exists(checkpoint_name):
			shutil.rmtree(checkpoint_name)
		if LEARNING_ALG == 0:
			tf.saved_model.simple_save(model.sess, checkpoint_name, inputs={"obs": model.act_model.obs_ph},
									outputs={"action": model.action_ph})
		elif LEARNING_ALG == 2:
			tf.saved_model.simple_save(model.sess, checkpoint_name, inputs={"obs": model.policy_tf.obs_ph},
									outputs={"action": model.policy_tf.deterministic_policy})
									# outputs={"action": model.actions_ph})
		else:
			raise NotImplementedError

	# save observation and action space 
	venv = env #env.env.env.env
	#quadruped_env = env.env.env.env.venv.envs[0].env
	print('mean', venv.obs_rms.mean)
	print('var ', venv.obs_rms.var)
	vecnorm_arr = np.vstack((venv.obs_rms.mean, 
							venv.obs_rms.var,
							venv.observation_space.high,
							venv.observation_space.low))
	# act_space_arr = np.vstack((
	# 						np.concatenate((quadruped_env._robot_config.IK_POS_UPP_LEG,quadruped_env._robot_config.MAX_NORMAL_FORCE)),
	# 						np.concatenate((quadruped_env._robot_config.IK_POS_LOW_LEG,quadruped_env._robot_config.MIN_NORMAL_FORCE))
	# 						))
	# np.savetxt('./exported/vecnorm_params.csv',vecnorm_arr,delimiter=',')
	# np.savetxt('./exported/action_space.csv', act_space_arr, delimiter=',')
	np.savetxt(os.path.join(write_out_path, 'my_model/vecnorm_params.csv'), vecnorm_arr,delimiter=',')
	np.savetxt(os.path.join(write_out_path, 'my_model2/vecnorm_params.csv'),vecnorm_arr,delimiter=',')
	#np.savetxt(write_out_path + 'action_space.csv', act_space_arr, delimiter=',')

	#sys.exit()
	#from ray.tune.trial import ExportFormat

	#agent.export_policy_model('./temp69')
	#agent.export_model([ExportFormat.CHECKPOINT, ExportFormat.MODEL],'./temp5')
	# policy = agent.get_policy()
	# print(policy)
	# print(policy.get_session())
	# print('variables', policy.variables())
	# #agent.export_policy_checkpoint('./temp11')
	# #with policy.get_session():
	i = tf.initializers.global_variables()
	with model.sess as sess:
		saver = tf.train.Saver()#policy.variables())#tf.global_variables())
		# saver.save(sess, './exported/my_model')
		# tf.train.write_graph(sess.graph, '.', './exported/graph.pb', as_text=False)
		saver.save(sess, os.path.join(write_out_path, 'my_model2/model'))
		tf.train.write_graph(sess.graph, '.', os.path.join(write_out_path, 'my_model2/graph.pb'), as_text=False)

#sys.exit()
x, y, z = [], [], []
real_x, real_y, real_z = [], [], []
roll, pitch, yaw = [], [], []
real_roll, real_pitch, real_yaw = [], [], []


rollout_num = 0

print(env.envs[0].env.opt_traj_filenames)
try:
	MAX_NUM_ROLLOUTS = len(env.envs[0].env.opt_traj_filenames)
except:
	print('setting max rollouts to default')
MAX_NUM_ROLLOUTS = 10

outside_torque_limits_counter = 0
obs = env.reset()
print('base',np.asarray(env.envs[0].env._robot.GetBasePosition()))
# for i in range(200):
while rollout_num < MAX_NUM_ROLLOUTS:
	#print('TIME',env.envs[0].env.get_sim_time())
	action, _states = model.predict(obs)
	#action = obs
	print('*'*80,'\n Observation:',obs.shape)
	nicePrint2D(obs)
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
	torque_ranges = np.asarray( [33.5] * 12 )#np.asarray( [44,44,55] * 4 )
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

	obs, rewards, dones, info = env.step(action)


	xyz = env.envs[0].env.save_traj_base_xyz
	x = xyz[0::3]
	y = xyz[1::3]
	z = xyz[2::3]
	try:
		rpy = env.envs[0].env.save_traj_base_rpy
		roll = rpy[0::3]
		pitch = rpy[1::3]
		yaw = rpy[2::3] 
	except:
		pass
	#print('x',xyz)
	#print('Fn', Fn)
	real_xyz = env.envs[0].env.save_real_base_xyz
	real_x = real_xyz[0::3]
	real_y = real_xyz[1::3]
	real_z = real_xyz[2::3]
	try:
		real_rpy = env.envs[0].env.save_real_base_rpy
		real_roll = real_rpy[0::3]
		real_pitch = real_rpy[1::3]
		real_yaw = real_rpy[2::3]
	except:
		pass
	# real_x.extend(real_xyz[0::3])
	# real_y.extend(real_xyz[1::3])
	# real_z.extend(real_xyz[2::3])

	if dones:
		#break
		#obs = env.reset()
		print('des: x', x[-1], 'z', z[-1], 'pitch', pitch[-1])
		print('act x', real_x[-1], 'z',real_z[-1], 'pitch', real_pitch[-1])
		print('*'*80, '\nreset\n','*'*80)
		rollout_num += 1
		print('base',np.asarray(env.envs[0].env._robot.GetBasePosition()))

	#if ROBOT > 1: # minitaur or rex
	#	env.render()
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

plt.plot(t,x, label='des x')
plt.plot(t,y, label='des y')
plt.plot(t,z, label='des z')
plt.plot(t,real_x, label='real_x')
plt.plot(t,real_y, label='real_y')
plt.plot(t,real_z, label='real_z')
plt.legend()
plt.show()

plt.plot(t,roll, label='des roll')
plt.plot(t,pitch, label='des pitch')
plt.plot(t,yaw, label='des yaw')
plt.plot(t,real_roll, label='real_roll')
plt.plot(t,real_pitch, label='real_pitch')
plt.plot(t,real_yaw, label='real_yaw')
plt.legend()
plt.show()

# plt.plot(t,Fn[:len(x)], label='Fn')
# plt.plot(t,real_Fn[:len(x)], label='real_Fn')
# plt.legend()
# plt.show()