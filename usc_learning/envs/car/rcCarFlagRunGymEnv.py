import os,  inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
currentdir = os.path.dirname(currentdir)

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)

import sys

import time
import datetime 
import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
from usc_learning.envs.car import rc_car
from pkg_resources import parse_version
import pdb

URDF_ROOT = os.path.join(currentdir, 'pyb_data/')
# URDF_ROOT = os.path.join(parentdir, 'pyb_data/')
print('URDF ROOT', URDF_ROOT)
URDF_FILENAME = 'racecar/racecar_differential_me.urdf'
os.sys.path.insert(0,URDF_ROOT)

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
VIDEO_LOG_DIRECTORY = 'videos/rcCarFlagRun/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")


""" Flag run environment for pybullet. Goal is to reach a bunch of random flags, like in Humanoid Flag Run. 

Action space: steering angle, desired body velocity
Observation space: see get_real_obs()
"""


def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	#print(vector)
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2' """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rot_mat(theta):
	return np.array([ [np.cos(theta), -np.sin(theta) ], [np.sin(theta), np.cos(theta)] ])


class RcCarFlagRunGymEnv(gym.Env):
	metadata = {
			'render.modes': ['human', 'rgb_array'],
			'video.frames_per_second' : 50
	}
	def __init__(self,
				 urdfRoot=URDF_ROOT,
				 time_step=0.01,
				 action_repeat=5, 
				 render=False,
				 record_video=False,
				 hard_reset=False): 

		self._urdfRoot = urdfRoot
		self._timeStep = time_step
		self._actionRepeat = action_repeat
		self._render = render
		self._is_record_video = record_video

		self._observation = []
		self._envStepCounter = 0

		if self._render:
			self._p = bc.BulletClient(connection_mode=pybullet.GUI)
			self._configure_visualizer()
		else:
			self._p = bc.BulletClient()

		self.seed()
		observationDim = 7
		observation_high = np.ones(observationDim) * 100 

		action_dim = 2
		self._action_bound = 1
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
		self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)

		self.goal = []
		self.dist_to_goal_threshold = 0.2 # threshold to consider we "got to goal"

		self.num_lookahead_steps = 1
		self.videoLogID = None
		self._hard_reset = True
		self.reset()
		self._hard_reset = hard_reset
		self.display_alg = 'RL'
		self.compute_returns = 0
		self.orig_dist2Goal = 0

	######################################################################################
	# Reset 
	######################################################################################
	def set_new_goal(self):
		"""Goal is a random location in 10x10m box. """
		dist = 1 + 4.*np.random.random()
		ang = 2.*3.1415925438*np.random.random()

		ballx = dist * math.sin(ang)
		bally = dist * math.cos(ang)
		ballz = 0.1

		self.goal = [ballx,bally,ballz]
		#self.optObj.setJ(xgoal=ballx,ygoal=bally)

		return ballx, bally, ballz

	def get_goal(self):
		""" Return current goal. """
		return self.goal

	def random_reset(self):
		""" 
		Set random position and velocity for car, to help randomize opt runner for supervised learning, 
		which is only being shown specific observation space sub-space
		"""
		dist = 1 + 4.*np.random.random()
		ang = 2.*3.1415925438*np.random.random()
		x = dist * math.sin(ang)
		y = dist * math.cos(ang)
		z = self.starting_carpos[2]

		yaw = 2.*3.1415925438*np.random.random() - 3.1415925438
		orn = self._p.getQuaternionFromEuler([0,0,yaw])

		linvel_max = 1.
		vx = 2*linvel_max*np.random.random() - linvel_max
		vy = 2*linvel_max*np.random.random() - linvel_max

		angvel_max = 1.
		wz = 2*angvel_max*np.random.random() - angvel_max

		self._p.resetBasePositionAndOrientation(self._racecar.racecarUniqueId, [x,y,z], orn)
		self._p.resetBaseVelocity(self._racecar.racecarUniqueId,[vx, vy, 0], [0, 0, wz])


	def reset(self):
		if self._hard_reset:
			self._p.resetSimulation()
			#self._p.setPhysicsEngineParameter(numSolverIterations=300)
			self._p.setTimeStep(self._timeStep)

			print('\n\nurdf root', self._urdfRoot)
			print('full path', os.path.join(self._urdfRoot,"plane.urdf"))

			self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))
			self._p.setGravity(0,0,-10)
			self._racecar = rc_car.RcCar(self._p,urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

			for i in range(100):
				self._p.stepSimulation()

			self.starting_carpos,self.starting_carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

		else:
			# just reset car body to 0
			self._p.resetBasePositionAndOrientation(self._racecar.racecarUniqueId,self.starting_carpos,self.starting_carorn)
			self._p.resetBaseVelocity(self._racecar.racecarUniqueId,[0, 0, 0], [0, 0, 0])

		self._envStepCounter = 0

		ballx, bally, ballz = self.set_new_goal()
		self.compute_returns = 0
		self.orig_dist2Goal = self.dist2Goal()

		if self._is_record_video:
			self.recordVideoHelper()

		self._observation = self.getExtendedObservationGoal()
		return np.array(self._observation)


	######################################################################################
	# Observation related, and also to get relevant info for optimization 
	######################################################################################
	def getExtendedObservationGoal(self):
		"""Get observation to goal. """
		self._observation = []
		self._observation.extend(self.get_real_obs())
		return self._observation

	def get_real_obs(self):
		"""Get observation, which are:
		-distance from COM to goal
		-angle between current heading and goal
		-steering angle
		-body velocity in global x direction
		-body velocity in global y direction
		-yaw rate
		-steering angle velocity (how quickly turning about z)

		Note: not using getBasePositionAndOrientation b/c not at geometric center of body.
			Instead, using a dummy link at the geometric center.
		"""
		_, _, _, _, pos, orn, linVel, angVel = self._p.getLinkState(self._racecar.racecarUniqueId, 20, 
											computeLinkVelocity=True, computeForwardKinematics=True)

		pos = np.asarray(pos)
		roll, pitch, yaw = self._p.getEulerFromQuaternion(orn)
		goal_vec = np.array(self.goal[0:2])
		pos[2] = 0

		distCurrPos2Goal, ang2Goal = self.getDistAndAngle2Goal(pos[0],pos[1],yaw,goal_vec)

		# include information on thf, velocities in x and y directions ( linVel[0:2] )
		steering_jointStates =  self._p.getJointStates(self._racecar.racecarUniqueId, self._racecar.steeringLinks)
		steering_jointPositions = np.array([x[0] for x in steering_jointStates])
		steering_jointVelocities = np.array([x[1] for x in steering_jointStates]) 

		thf = np.mean(steering_jointPositions)
		dthf = np.mean(steering_jointVelocities)
		dxb = linVel[0]
		dyb = linVel[1]
		dthb = angVel[2]

		return [ distCurrPos2Goal, ang2Goal, thf, dxb, dyb, dthb, dthf ]

	def getDistAndAngle2Goal(self, xb, yb, thb, goal):
		"""Get distance and angle from current heading to goal."""
		yaw = thb
		goal_vec = goal
		pos = np.array([xb, yb])

		# distance between current pos and goal
		distCurrPos2Goal = np.linalg.norm( pos[0:2] - goal_vec )

		# angle to goal (from current heading)
		body_dir_vec = np.matmul( rot_mat(yaw), np.array([[1],[0]]) )
		body_goal_vec = goal_vec - pos[0:2]
		body_dir_vec = body_dir_vec.reshape(2,)
		body_goal_vec = body_goal_vec.reshape(2,)

		Vn = unit_vector( np.array([0,0,1]) )
		c = np.cross( np.hstack([body_dir_vec,0]), np.hstack([body_goal_vec,0])  )
		angle = angle_between(body_dir_vec, body_goal_vec)
		angle = angle * np.sign( np.dot( Vn , c ) )

		return distCurrPos2Goal, angle


	def dist2Goal(self):
		"""Calculate distance from current body center to goal location. """
		_,_,_,_, base_pos, orn = self._p.getLinkState(self._racecar.racecarUniqueId, 20, computeForwardKinematics=True)
		base_pos = np.array(base_pos[0:2])
		goal = np.array(self.goal[0:2])
		dist_to_goal = np.linalg.norm(  base_pos - goal )
		return dist_to_goal

	def get_optimization_state(self):
		""" Get state w0 for optimization. """
		return self._racecar.get_w0_state()

	def step(self, action):
		"""Scale action from [-1,1], step sim, etc."""
		prev_dist_to_goal = self.dist2Goal()

		if self._render:
			self._render_step_helper()
		# clip and scale (not really needed)
		realaction = np.clip(action, self.action_space.low, self.action_space.high)
		#realaction = realaction * (self.action_space.high - self.action_space.low) + self.action_space.low
		
		for _ in range(self._actionRepeat):
			self._racecar.applyAction(realaction)
			self._p.stepSimulation()
			self._observation = self.getExtendedObservationGoal()
			after_dist_to_goal = self.dist2Goal()
			if self._termination():# or (after_dist_to_goal < self.dist_to_goal_threshold ):
				break
			if self._render:
				time.sleep(self._timeStep)
		
			self._envStepCounter += 1

		# current car pos and orn
		_, _, _, _, base_pos, orn = self._p.getLinkState(self._racecar.racecarUniqueId, 20, computeForwardKinematics=True)

		reward = ( prev_dist_to_goal - after_dist_to_goal ) #/ (self._actionRepeat * self._timeStep)
		done = self._termination() 
		# set new goal if we are within the threshold for reaching this one
		if (after_dist_to_goal < self.dist_to_goal_threshold ):
			self.set_new_goal()
		# keep track of returns
		self.compute_returns = self.compute_returns + reward

		if done:
			print("counter", self._envStepCounter,"returns",self.compute_returns, "orig_dist2Goal",self.orig_dist2Goal, "goal", self.goal, "end_pos", base_pos)

		return np.array(self._observation), reward, done, {'goal':self.goal,'car_state':self.get_optimization_state()}

	def _termination(self):
		#return self._envStepCounter>1000
		return self._envStepCounter>1000

	######################################################################################
	# Render, videos, and misc
	######################################################################################
	def startRecordingVideo(self,name):
		self.videoLogID = self._p.startStateLogging(
							self._p.STATE_LOGGING_VIDEO_MP4, 
							name)

	def stopRecordingVideo(self):
		self._p.stopStateLogging(self.videoLogID)

	def close(self):
		if self._is_record_video:
			self.stopRecordingVideo()
		self._p.disconnect()

	def recordVideoHelper(self):
		""" Helper to record video, if not already, or end and start a new one """
		# If no ID, this is the first video, so make a directory and start logging
		if self.videoLogID == None:
			#directoryName = VIDEO_LOG_DIRECTORY + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")
			directoryName = VIDEO_LOG_DIRECTORY
			assert isinstance(directoryName, str)
			os.makedirs(directoryName, exist_ok=True)
			self.videoDirectory = directoryName
		else:
			# stop recording and record a new one
			self.stopRecordingVideo()

		output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"
		logID = self.startRecordingVideo(output_video_filename)
		self.videoLogID = logID

	def _render_step_helper(self):
		""" Helper to configure the visualizer camera during step(). """
		basePos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
		self.display_alg = 'RL'
		try:
			self.vis_goal = self._p.addUserDebugText(text='X',textPosition=self.goal,textSize=2,lifeTime=0,textColorRGB=[1,0,0],replaceItemUniqueId=self.vis_goal)
			self.vis_alg = self._p.addUserDebugText(text=self.display_alg, textPosition=[self.goal[0]+.5, self.goal[1], .3], textSize=2,lifeTime=0,textColorRGB=[0,0,1],replaceItemUniqueId=self.vis_alg)
		except:
			self.vis_goal = self._p.addUserDebugText(text='X',textPosition=self.goal,textSize=2,lifeTime=0,textColorRGB=[1,0,0])
			self.vis_alg  = self._p.addUserDebugText(text=self.display_alg, textPosition=[self.goal[0]+.5, self.goal[1], .3], textSize=2,lifeTime=0,textColorRGB=[0,0,1])

		#self._p.resetDebugVisualizerCamera(5, 0, -89.999,  basePos)
		self._p.resetDebugVisualizerCamera(6, 0, -89.999,  [0,0,0])


	def render(self, mode='human', close=False):
		if mode != "rgb_array":
			return np.array([])
		base_pos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
				cameraTargetPosition=base_pos,
				distance=self._cam_dist,
				yaw=self._cam_yaw,
				pitch=self._cam_pitch,
				roll=0,
				upAxisIndex=2)
		proj_matrix = self._p.computeProjectionMatrixFOV(
				fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
				nearVal=0.1, farVal=100.0)
		(_, _, px, _, _) = self._p.getCameraImage(
				width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
				projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array


	def _configure_visualizer(self):
		""" Remove all visualizer borders, and zoom in """
		# default rendering options
		self._render_width = 960
		self._render_height = 720
		# self._camera_dist = 1
		# self._camera_pitch = -30
		# self._camera_yaw = 0
		self._cam_dist = 1.0
		self._cam_yaw = 0
		self._cam_pitch = -30
		# get rid of random visualizer things
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI,0)

	def __del__(self):
		self._p = 0

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	if parse_version(gym.__version__) < parse_version('0.9.6'):
		_render = render
		_reset = reset
		_seed = seed
		_step = step
