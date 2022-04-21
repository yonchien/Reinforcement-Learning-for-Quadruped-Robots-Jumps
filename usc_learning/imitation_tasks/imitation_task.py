import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
import numpy as np
import time

from usc_learning.envs.aliengo import AlienGo 
from usc_learning.envs.aliengo2D import AlienGo2D
from usc_learning.imitation_tasks.traj_motion_data import TrajTaskMotionData

""" Implemented task environemnts """
TASK_ENVS = ["STANDUP", "FWD_LOCOMOTION"]

""" See available control modes in aliengo.py, currently:
CONTROL_MODES = [ "TORQUE", "PD", "IK", "FORCE", "IMPEDANCE" ]
"""
class ImitationTaskAlienGoGymEnv(gym.Env):
	""" The gym environment interface to the AlienGo robot (in aliengo.py) 
	for imitating jumping maneuvers produce from the trajectory optimization framework. """

	def __init__(self,
				 control_mode="PD",
				 task_env="STANDUP",
				 traj_filename="data/jumpingFull2.csv",
				 env2D=True,
				 actionRepeat=10,
				 timeStep=0.002,
				 isrender=False):
		""" Check other arguments
		- imitation task file (or separate env?)
		- how often to update the imitation timeStep
		- how often to update the control loop
		- how to imitate? (i.e. add torque on top of existing taus from opt, positions, PD gains, etc.)
		- action repeats
		- control modes?

		"""
		self._control_mode = control_mode
		self._TASK_ENV = task_env
		self._action_repeat = actionRepeat
		self._time_step = timeStep
		self._is_render = isrender
		self._env2D = env2D

		# desired trajectory to track
		self._traj_task = TrajTaskMotionData(filename=traj_filename)

		if self._is_render:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
			#self._pybullet_client = p.connect(p.GUI)
		else:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
			#self._pybullet_client = p.connect(p.DIRECT)

		if not env2D:
			self._robot = AlienGo(pybullet_client=self._pybullet_client,
									isGymInterface=True,
									control_mode=self._control_mode,
									time_step=self._time_step,
									actionRepeat=self._action_repeat)

			# set action space
			if self._control_mode in ["TORQUE","PD","IK"]: # if Torque, PD, or IK
				action_dim = 12
			elif self._control_mode == "FORCE":
				action_dim = 4
			else:
				""" IMPEDANCE (or others) TODO """
				action_dim = 12 

		else:
			self._robot = AlienGo2D(pybullet_client=self._pybullet_client,
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
				raise ValueError("action space TBD")
				action_dim = 12 


		self.reset()

		# Observation and action spaces
		obs_dim = len(self._get_obs())
		#obs_high = 100*np.ones(obs_dim) # very conservative
		# takes care of either 2D or normal env
		obs_low, obs_high = self._robot._defaultObsLimits()

		if self._env2D:
			obs_high = np.concatenate( (obs_high, 
										np.amax(self._traj_task.full_state2D, axis=1) ))
			obs_low  = np.concatenate( (obs_low, 
										np.amin(self._traj_task.full_state2D, axis=1) ))
		else:
			obs_high = np.concatenate( (obs_high, 
										np.amax(self._traj_task.full_state, axis=1) ))
			obs_low  = np.concatenate( (obs_low, 
										np.amin(self._traj_task.full_state, axis=1) ))

		action_high = np.ones(action_dim)
		self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
		self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

		self.seed()
		self.setup_render()


	def _get_obs(self):
		""" Get observation for imitation task. 
		Concatenate next desired observation(s) (next state(s) in traj)
		- TODO: add input to determine how many future trajectory snapshots to include in observation 
		(see google paper, which has something like 1 dt ahead, 5 dt ahead, 0.33 seconds ahead, 1 second ahead)
		"""
		curr_time = self.getSimTime()
		traj_dt = self._traj_task.dt
		traj_duration = self._traj_task.get_duration()
		# times at which to query opt traj data
		obs_times = max(curr_time+traj_dt, traj_duration)
		raise ValueError('[DEBUG] should be min? otherwise always using traj_duration (6/18/2020)')

		if self._env2D: 
			robot_state = self._robot.GetFullState2D()
		else:
			robot_state = self._robot.GetFullState()
			
		# add desired states to observation 
		traj_desired_state = self._traj_task.get_state_at_time(obs_times,env2D=self._env2D)
		obs = np.hstack(( robot_state, traj_desired_state ))

		return obs

	def step(self,action):
		""" Step robot. 

		TODO: add action to tau from optimization
		"""
		# action will be processed and scaled by robot class internal _scale functions
		self._robot.Step(action)
		#terminate = self._robot._terminate()
		#reward = self._reward()
		reward, terminate = self._reward_and_terminate()

		return self._get_obs(), reward, terminate, {}

	def _reward(self):
		""" Get reward. 
		weight 
		-base pos/orientation difference,
		-joint angles/ vels
		-distance to goal?
		-survival bonus?
		-keep torques close to default?
		"""
		if self._TASK_ENV == "STANDUP":
			return self._rewardStandUp()
		if self._TASK_ENV == "FWD_LOCOMOTION":
			return self._rewardStandUp()
		else:
			raise ValueError("This task mode not implemented yet.")

	def _reward_and_terminate(self):
		""" Get reward and terminate condition for imitation task.

		Reward is a weighted sum of the errors in:
			-base pos/orn/linvel/angvel
			-joint pos/vel 
		Consider adding cost/reward for:
			-distance to goal
			-survival bonus
			-torques offset from desired traj? anneal over time to include offsets?

		Terminate if:
			-diverge too much from trajectory
			-trajectory is done
		"""
		curr_time = self.getSimTime()

		# desired trajectory state at this time
		traj_base_pos, traj_base_orn, traj_base_linvel, \
		traj_base_angvel, traj_joint_pos, traj_joint_vel = self.get_traj_state_at_time(curr_time)

		# current robot state
		curr_base_pos, curr_base_orn, curr_base_linvel, \
		curr_base_angvel, curr_joint_pos, curr_joint_vel = self.get_robot_state()

		self._base_pos_weight = 1
		self._base_orn_weight = .5
		self._base_linvel_weight = 0
		self._base_angvel_weight = 0
		self._joint_pos_weight = 0
		self._joint_vel_weight = 0

		reward = self._base_pos_weight * np.linalg.norm( traj_base_pos - curr_base_pos ) \
					+ self._base_orn_weight * np.linalg.norm( traj_base_orn - curr_base_orn ) \
					+ self._base_linvel_weight * np.linalg.norm( traj_base_linvel - curr_base_linvel ) \
					+ self._base_angvel_weight * np.linalg.norm( traj_base_angvel - curr_base_angvel ) \
					+ self._joint_pos_weight * np.linalg.norm( traj_joint_pos - curr_joint_pos ) \
					+ self._joint_vel_weight * np.linalg.norm( traj_joint_vel - curr_joint_vel )

		reward = 1 - reward # survival bonus
		terminate = False
		#print(traj_base_pos,traj_base_orn)
		#print(curr_base_pos,curr_base_orn)
		# terminate if diverge too much from trajectory
		if np.linalg.norm( np.concatenate((traj_base_pos,traj_base_orn)) \
						 - np.concatenate((curr_base_pos,curr_base_orn)) ) > 0.5:
			terminate = True

		# terminate if trajectory is over
		if curr_time > self._traj_task.get_duration():
			terminate = True

		return reward, terminate

	def reset(self):
		""" Reset the robot position. """
		self._robot.Reset()
		# move robot to traj start and settle
		base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel = self.get_traj_state_at_time(0)
		init_base = np.concatenate((base_pos,base_orn))
		init_joints = np.concatenate((joint_pos,joint_vel))
		self._robot.SetPoseAndSettle(init_base,init_joints)
		return self._get_obs()

	##############################################################################
	## Task specific 
	##############################################################################

	def _rewardStandUp(self, desHeight=0.4):
		""" Calculate reward for standup task """
		last_z = self._robot._last_base_pos[2]
		curr_z = self._robot.GetBasePosition()[2]

		reward = -100 * ( np.linalg.norm(curr_z - desHeight) - np.linalg.norm(last_z - desHeight) )
		return reward

	def _rewardFwd(self):
		""" Calculate reward for forward locomotion """
		last_x = self._robot._last_base_pos[0]
		curr_x = self._robot.GetBasePosition()[0]
		# reward for velocity minus control effort
		reward = (curr_x -last_x)/(self._time_step*self._action_repeat) - 1e-6*np.linalg.norm(self._robot.GetAppliedTorques())
		return reward

	def _get_observation_standup(self):
		""" get observation for standup task """
		# default is robot.DefaultStandUpObs()
		return self._robot.DefaultStandUpObs()

	##############################################################################
	## Imitation helpers 
	##############################################################################
	def getSimTime(self):
		""" Get current simulation time from pybullet robot. """
		return self._robot.GetSimTime()

	def get_traj_state_at_time(self,t):
		""" Get the traj state at time t. 
		Return:
		-base_pos
		-base_orn
		-base_linvel
		-base_angvel
		-joint_pos
		-joint_vel
		"""
		traj_state = self._traj_task.get_state_at_time(t)
		base_pos = self._traj_task.get_base_pos_from_state(traj_state)#,env2D=self._env2D)
		base_orn = self._traj_task.get_base_orn_from_state(traj_state)#,env2D=self._env2D)
		base_linvel = self._traj_task.get_base_linvel_from_state(traj_state)#,env2D=self._env2D)
		base_angvel = self._traj_task.get_base_angvel_from_state(traj_state)#,env2D=self._env2D)

		joint_pos = self._traj_task.get_joint_pos_from_state(traj_state)#,env2D=self._env2D)
		joint_vel = self._traj_task.get_joint_vel_from_state(traj_state)#,env2D=self._env2D)
		
		return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel 

	def get_robot_state(self):
		""" Get the current robot state. 
		Return:
		-base_pos
		-base_orn
		-base_linvel
		-base_angvel
		-joint_pos
		-joint_vel
		"""
		# if self._env2D:
		# 	base_state = self._robot.GetBaseState2D()
		# 	base_pos = base_state[0:2]
		# 	base_orn = [base_state[2]]
		# 	base_linvel = base_state[3:5]
		# 	base_angvel = [base_state[5]]
		# 	joint_pos, joint_vel = np.split(self._robot.GetJointState2D(), 2)
		# else:
		base_pos = self._robot.GetBasePosition()
		base_orn = self._robot.GetBaseOrientationRollPitchYaw()
		base_linvel = self._robot.GetBaseLinearVelocity()
		base_angvel = self._robot.GetBaseAngularVelocity()
		joint_pos, joint_vel = np.split(self._robot.GetJointState(), 2)
		#joint_pos = self._robot.GetJointAngles()
		#joint_vel = self._robot.GetJointVelocities()

		return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel 


	##############################################################################
	## render, there is a bug when using stable baselines (maybe from wrappers)
	## cannot access visualizer
	##############################################################################
	def setup_render(self):
		""" Default rendering options. """
		# default rendering options
		self._render_width = 480
		self._render_height = 360
		self._camera_dist = 1
		self._camera_pitch = -30
		self._camera_yaw = 0

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
		return rgb_array

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