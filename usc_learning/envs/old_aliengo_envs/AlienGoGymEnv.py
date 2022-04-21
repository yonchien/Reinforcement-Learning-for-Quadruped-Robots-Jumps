import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
import numpy as np
import time

from usc_learning.envs import aliengo, aliengo2D
"""
Notes:
careful with negative rewards and termination conditions, 
agent can learn to end early to stop accumulating negative rewards

"""

""" Implemented task environemnts """
TASK_ENVS = ["STANDUP", "FWD_LOCOMOTION", "STAYDOWN"]

""" See available control modes in aliengo.py, currently:
CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD",  "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE" ]
"""
class AlienGoGymEnv(gym.Env):
	""" The gym environment interface to the AlienGo robot (in aliengo.py) """

	def __init__(self,
				 control_mode="PD_NO_HIPS",
				 task_env="STANDUP",
				 env2D=False,
				 actionRepeat=5,
				 timeStep=0.005,
				 isrender=False):
		""" Check other arguments
		- imitation task file (or separate env?)
		- action repeats
		- control modes?

		"""
		self._control_mode = control_mode
		self._TASK_ENV = task_env
		self._action_repeat = actionRepeat
		self._time_step = timeStep
		self._is_render = isrender
		self._env2d = env2D
		if self._is_render:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
			p.configureDebugVisualizer(
								p.COV_ENABLE_GUI,
								True)
			self.configureVisualizer()
		else:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)

		if not env2D:
			self._robot = aliengo.AlienGo(pybullet_client=self._pybullet_client,
											isGymInterface=True,
											control_mode=self._control_mode,
											time_step=self._time_step,
											actionRepeat=self._action_repeat)

			# set action space
			if self._control_mode in ["TORQUE","PD","IK"]: # if Torque, PD, or IK
				action_dim = 12
			elif self._control_mode in ["TORQUE_NO_HIPS", "PD_NO_HIPS"]:
				action_dim = 8
			elif self._control_mode == "FORCE":
				action_dim = 4
			else:
				""" IMPEDANCE (or others) TODO """
				action_dim = 12 

		else:
			self._robot = aliengo2D.AlienGo2D(pybullet_client=self._pybullet_client,
											isGymInterface=True,
											control_mode=self._control_mode,
											time_step=self._time_step,
											actionRepeat=self._action_repeat)

			# set 2D action sapces
			if self._control_mode in ["TORQUE","PD"]: # if Torque, PD
				action_dim = 4
			elif self._control_mode == "FORCE":
				action_dim = 2
			else:
				""" IMPEDANCE (or others) TODO """
				raise ValueError("action space TBD for control mode", self._control_mode)
				action_dim = 12 


		obs_dim = len(self._get_obs())
		#obs_high = 100*np.ones(obs_dim) # very conservative
		# takes care of either 2D or normal env
		obs_low, obs_high = self._robot._defaultObsLimits()
		action_high = np.ones(action_dim)
		self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
		self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

		self.seed()
		self._MAX_EP_LEN = 5 # seconds, arbitrary 
		

		# # Reward weights
		# self._relZdist_w = 100
		# self._absZdist_w = 100
		# self._ori_w = 100
		# self._termination_w = 20
		# self._energy_w = 0.0001
		# self._shake_w = 0.1
		# self._shake_vel_w = 0.01
		# self._shake_orn_w = 0.0001

		# Reward weights
		self._relZdist_w = 200
		self._absZdist_w = 0
		self._ori_w = 20
		self._termination_w = 200
		self._energy_w = 0
		self._shake_w = 20
		self._shake_vel_w = 0
		self._shake_orn_w = 0

		#termination threshold
		self._tip_threshold = 1


	def _get_obs(self):
		""" Get observation, depending on task. """
		if self._TASK_ENV == "STANDUP" or self._TASK_ENV == "FWD_LOCOMOTION" or self._TASK_ENV == "STAYDOWN":
			return self._get_observation_standup()
		else:
			raise ValueError("This task mode not implemented yet.")

	def step(self,action):
		""" Step robot """
		# action will be processed and scaled by robot class internal _scale functions
		#print('getting action')
		#print(action)
		self._robot.Step(action)
		terminate = self._robot._terminate(self._tip_threshold) or self._robot.IsBodyInContact()
		terminate = terminate or self.getSimTime() > self._MAX_EP_LEN #False
		reward = self._reward()
		# if terminate:
		# 	reward -= self._termination_w

		return self._get_obs(), reward, terminate, {}

	def _reward(self):
		""" Get reward. """
		if self._TASK_ENV == "STANDUP":
			return self._rewardStandUp()
		elif self._TASK_ENV == "FWD_LOCOMOTION":
			return self._rewardFwd()
		elif self._TASK_ENV == "STAYDOWN":
			return self._rewardStayDown()
		else:
			raise ValueError("This task mode not implemented yet.")

	def reset(self):
		""" Reset the robot position. """
		self._robot.Reset()
		return self._get_obs()

	def close(self):
		""" Close the Aliengo env """
		self._robot.close()
	##############################################################################
	## Task specific 
	##############################################################################
	def _rewardStandUpNotes(self):
		""" old notes -g """
		last_z = self._robot._last_base_pos[2]
		curr_z = self._robot.GetBasePosition()[2]
		curr_orn = self._robot.GetBaseOrientationRollPitchYaw()
		curr_quat_orn = np.asarray(self._robot.GetBaseOrientation())
		
		#reward = -100 * ( np.linalg.norm(curr_z - desHeight) - np.linalg.norm(last_z - desHeight) )
		#reward = 10* (1 - abs(curr_z - desHeight) )
		reward = (desHeight - abs(curr_z - desHeight) ) - 0.1*np.linalg.norm(curr_quat_orn - [0,0,0,1])
		reward = - abs(curr_z - desHeight) - 0.1*np.linalg.norm(curr_quat_orn - [0,0,0,1])


		x = abs(curr_z - desHeight)
		sigma = 0.1
		# Kx = - 1 / (np.exp(x) + 2 + np.exp(-x))
		# reward = Kx
		#print('rew', 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x/sigma)**2), np.exp(-0.5 * (x/sigma)**2), 10*(curr_z - desHeight)**2)

		reward = np.exp(-0.5 * (x/sigma)**2) - 0.1*np.linalg.norm(curr_quat_orn - [0,0,0,1])

		joints_vel_penalty = np.sum( self._robot.GetJointVelocities()**2 )
		body_vel_penalty = np.sum( self._robot.GetBaseLinearVelocity()**2 ) + np.sum( self._robot.GetBaseAngularVelocity()**2 )
		torque_penalty = np.sum( self._robot.GetAppliedTorques()**2 )

		#reward = reward - 1e-4*joints_vel_penalty - 1e-4 * body_vel_penalty - 1e-3*curr_orn[0]**2 - 1e-3*curr_orn[1]**2
		return reward

	def _rewardStandUp(self):
		""" Calculate reward for standup task """
		reward = self._relZdist_w * self._rewardrelZDistance() \
				+ self._absZdist_w * self._rewardabsZDistance() \
				+ self._ori_w * self._rewardOrientation() \
				+ self._energy_w * self._rewardEnergy() \
				+ self._shake_w * self._rewardShake() \
				+ self._shake_vel_w * self._rewardShakeForVelocity() \
				+ self._shake_orn_w * self._rewardShakeForOrientation()
		return reward

	def _rewardStayDown(self):
		""" Simple test, keep body z as low as possible. """
		curr_orn = self._robot.GetBaseOrientationRollPitchYaw()
		return -self._robot.GetBasePosition()[2] - 0.1*curr_orn[0]**2 - 0.1*curr_orn[1]**2

	def _rewardFwd(self):
		""" Calculate reward for forward locomotion """
		last_x = self._robot._last_base_pos[0]
		curr_x = self._robot.GetBasePosition()[0]
		# reward for velocity minus control effort
		reward = (curr_x -last_x)/(self._time_step*self._action_repeat) - 1e-6*np.linalg.norm(self._robot.GetAppliedTorques())
		return reward

	def _rewardabsZDistance(self, desHeight=0.4):
		last_z = self._robot._last_base_pos[2]
		curr_z = self._robot.GetBasePosition()[2]
		reward = -np.linalg.norm(curr_z - desHeight)
		return reward

	def _rewardrelZDistance(self, desHeight=0.4):
		last_z = self._robot._last_base_pos[2]
		curr_z = self._robot.GetBasePosition()[2]
		reward = -( np.linalg.norm(curr_z - desHeight) - np.linalg.norm(last_z - desHeight) )
		return reward

	def _rewardOrientation(self,desOri = np.array([0,0,0])):
		orientation_reward = -(np.diff([self._robot.GetBaseOrientationRollPitchYaw(), desOri]) ** 2).sum()
		return orientation_reward

	
	def _rewardEnergy(self):
		reward = -np.abs(np.dot(self._robot.GetAppliedTorques(),
               self._robot.GetJointVelocities())) * self._time_step
		return reward

	def _rewardShake(self):
		# shake_reward = -np.abs(self._robot.GetBasePosition() - self._robot._last_base_pos)
		shake_reward = -(np.diff([self._robot.GetBasePosition(), self._robot._last_base_pos]) ** 2).sum()
		return shake_reward

	def _rewardShakeForVelocity(self):
		# shake_reward = -np.abs(self._robot.GetBasePosition() - self._robot._last_base_pos)
		shake_reward = -(np.diff([self._robot.GetBaseLinearVelocity(), self._robot._last_base_vel]) ** 2).sum()
		return shake_reward

	def _rewardShakeForOrientation(self):
		# shake_reward = -np.abs(self._robot.GetBasePosition() - self._robot._last_base_pos)
		shake_reward = -(np.diff([self._robot.GetBaseOrientationRollPitchYaw(), self._robot._last_base_orn]) ** 2).sum()
		return shake_reward

	def _get_observation_standup(self):
		""" get observation for standup task """
		# default is robot.DefaultStandUpObs()
		return self._robot.DefaultStandUpObs()

	def getSimTime(self):
		""" Get current simulation time from pybullet robot. """
		return self._robot.GetSimTime()

	##############################################################################
	## render, anything below here rarely needs to be looked at
	##############################################################################
	def render(self, mode='rgb_array'):
		""" render function, standard """
		if mode != 'rgb_array':
			raise ValueError('Unsupported render mode:{}'.format(mode))
		base_pos = self._robot.GetBasePosition()
		view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
				cameraTargetPosition=base_pos,
				distance=self._camera_dist,
				yaw=self._camera_yaw,
				pitch=self._camera_pitch,
				roll=0,
				upAxisIndex=2)
		proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
				fov=60,
				aspect=float(self._render_width) / self._render_height,
				nearVal=0.1,
				farVal=100.0)
		(_, _, px, _, _) = self._pybullet_client.getCameraImage(
				width=self._render_width,
				height=self._render_height,
				renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
				viewMatrix=view_matrix,
				projectionMatrix=proj_matrix)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		#self.render_good()
		return rgb_array


	def render_good(self):
		""" Remove all visualizer borders, and zoom in """
		# set camera
		vis_cam = self._pybullet_client.getDebugVisualizerCamera()
		pos, orn = self._robot.GetBasePosition(), self._robot.GetBaseOrientation()
		self._pybullet_client.resetDebugVisualizerCamera(vis_cam[10], vis_cam[8], vis_cam[9], pos)
		self._pybullet_client.resetDebugVisualizerCamera(1.5, vis_cam[8], vis_cam[9], [pos[0],pos[1],pos[2]])

	def configureVisualizer(self):
		""" Remove all visualizer borders, and zoom in """
		# default rendering options
		self._render_width = 480
		self._render_height = 360
		self._camera_dist = 1
		self._camera_pitch = -30
		self._camera_yaw = 0
		# get rid of random visualizer things
		self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
		self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
		self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
		self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

if __name__ == "__main__":
	# test out some functionalities
	env = AlienGoGymEnv(isrender=True)
	action_space_size = len(env.action_space.low)
	while True:
		action=2*np.random.random(action_space_size)-1
		obs, reward, done, info = env.step(action)
		if done:
			env.reset()
		time.sleep(0.05)