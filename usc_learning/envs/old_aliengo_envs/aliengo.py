""" This file implements the functionalities of an AlienGo robot in pybullet. 
	-Note: this is not intended to be used as Gym environment on its own, but should be called from RL or IL env.

	Organization:
	(1) Global variables, init
	(2) [Observations] - i.e. robot states, contacts, dynamics/kinematics, etc.
	(3) [Actions] - Input related - i.e. scaling, set torques, PD 
	(4) [Step] - (apply actions, step simulation)
	(5) [Reset] - set up quadruped/sim, reset, add noise to states, etc.
	(5) [I/O] - videos, viualization, etc.

	inspired from the minitaur/laikago pybullet classes
"""
import time, datetime
import numpy as np
import sys, os
import re

# remove importing pybullet afterwards? just for testing
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bullet_client

from usc_learning.utils.utils import nicePrint, nicePrint2D

""" URDF and strings """
#URDF_ROOT = pybullet_data.getDataPath()
#URDF_FILENAME = "aliengo/urdf/aliengo_rmBase.urdf"
#URDF_FILENAME = "aliengo/urdf/aliengo.urdf"
URDF_ROOT = os.path.join(os.path.dirname(__file__), 'pyb_data/')
# URDF_FILENAME = "aliengo/urdf/aliengo_hipLimits.urdf" # updated URDF values AND limiting hips in [-pi,pi]
URDF_FILENAME = "aliengo/urdf/aliengo_hipLimits_pi2.urdf"
VIDEO_LOG_DIRECTORY = 'videos/' 

_HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")

joint_strings = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
				 "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
				 "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
				 "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]

feet_strings = [b'FR_foot_fixed', b'FL_foot_fixed', b'RR_foot_fixed', b'RL_foot_fixed']

""" Joint position, velocity, and torque limits, and PD gains """
# thigh joints have no built in limit (what makes sense?)
UPPER_ANGLE_JOINT = [ 1.2217304764,  3.14, -0.645771823238] * 4
LOWER_ANGLE_JOINT = [-1.2217304764, -3.14, -2.77507351067 ] * 4 
""" For pi2 urdf"""
UPPER_ANGLE_JOINT = [ 1.2217304764,  1.57, -0.645771823238] * 4
LOWER_ANGLE_JOINT = [-1.2217304764, -1.57, -2.77507351067 ] * 4 


UPPER_ANGLE_JOINT = np.array(UPPER_ANGLE_JOINT)
LOWER_ANGLE_JOINT = np.array(LOWER_ANGLE_JOINT)

TORQUE_LIMITS = np.asarray( [44,44,55] * 4 )
VELOCITY_LIMITS = np.asarray( [20,20,16] * 4 )

""" NO HIPS """
UPPER_ANGLE_JOINT_NO_HIPS = [ 1.57, -0.645771823238 ] * 4 
LOWER_ANGLE_JOINT_NO_HIPS = [-1.57, -2.77507351067  ] * 4


UPPER_ANGLE_JOINT_NO_HIPS = np.array(UPPER_ANGLE_JOINT_NO_HIPS)
LOWER_ANGLE_JOINT_NO_HIPS = np.array(LOWER_ANGLE_JOINT_NO_HIPS)

TORQUE_LIMITS_NO_HIPS = np.asarray( [44,55] * 4 )
#VELOCITY_LIMITS_2D = np.asarray( [20,16] * 4 )
""" """

# JOINT_DIRECTIONS = np.array([ 1, 1, 1, 
# 							 -1, 1, 1, 
# 							  1, 1, 1, 
# 							 -1, 1, 1])

# hip positions
FR_hip = np.array([ 0.2399 , -0.051, 0])
FL_hip = np.array([ 0.2399 ,  0.051, 0])
RR_hip = np.array([-0.2399 , -0.051, 0])
RL_hip = np.array([-0.2399 ,  0.051, 0])

HIP_POSITIONS = [FR_hip, FL_hip, RR_hip, RL_hip]

# for better stability
# FR_hip_reset = np.array([ 0.2399 , -0.11, 0])
# FL_hip_reset = np.array([ 0.2399 ,  0.11, 0])
# RR_hip_reset = np.array([-0.2399 , -0.11, 0])
# RL_hip_reset = np.array([-0.2399 ,  0.11, 0])
hip_offset = 0.083
FL_thigh_from_hip = np.array([0,hip_offset,0])
FR_hip_reset = FR_hip - FL_thigh_from_hip 
FL_hip_reset = FL_hip + FL_thigh_from_hip 
RR_hip_reset = RR_hip - FL_thigh_from_hip 
RL_hip_reset = RL_hip + FL_thigh_from_hip 

HIP_POSITIONS_RESET = [FR_hip_reset, FL_hip_reset, RR_hip_reset, RL_hip_reset]

INIT_STAND_POSITION = [  0.03, 0.65, -1.2,
						-0.03, 0.65, -1.2,
						 0.03, 0.65, -1.2,
						-0.03, 0.65, -1.2] 

# from https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/body.cpp
""" TODO: the below gains need to be tuned 
"""
joint_kp_gains = [70,180,300]*4
joint_kd_gains = [3,8,15] * 4

#joint_kp_gains = [1]*12
#joint_kd_gains = [.2] * 12

#joint_kp_gains = [220.,220.,220.]*4
#joint_kd_gains = [5.,5.,5.] * 4

""" recommendation from ANYmal paper
https://arxiv.org/pdf/1901.08652.pdf
"""
joint_kp_gains = TORQUE_LIMITS / ( (UPPER_ANGLE_JOINT - LOWER_ANGLE_JOINT) / 2 )
joint_kd_gains = [.1] * 12

# joint_kp_gains = [0]*12
# joint_kd_gains = [0]*12

""" Current force control environment limits  """
MIN_NORMAL_FORCE = np.asarray([-80]*4)
MAX_NORMAL_FORCE = np.asarray([-30]*4)
MIN_XYZ_FORCE = np.asarray( [-20,-20,-60]*4)
MAX_XYZ_FORCE = np.asarray( [ 20, 20,-40]*4)


""" IK LIMITS BOX
TODO: verify these are reasonable 
"""
# scale possible area?
# for limb 0 (FR)
# x: (0.1, 0.5)
# y: (-0.2, -0.03)
# z: (-0.1, -0.5) ?
FR_low = [0.1, -0.2 , -0.5]
FR_upp = [0.5, -0.03, -0.1]

# WRONG max/min
# FL_low = [ FR_low[0], -FR_low[1], FR_low[2]]
# FL_upp = [ FR_upp[0], -FR_upp[1], FR_upp[2]]

# RR_low = [-FR_low[0],  FR_low[1], FR_low[2]]
# RR_upp = [-FR_upp[0],  FR_upp[1], FR_upp[2]]

# RL_low = [-FR_low[0], -FR_low[1], FR_low[2]]
# RL_upp = [-FR_upp[0], -FR_upp[1], FR_upp[2]]
FL_low = [ FR_low[0], -FR_upp[1], FR_low[2]]
FL_upp = [ FR_upp[0], -FR_low[1], FR_upp[2]]

RR_low = [-FR_upp[0],  FR_low[1], FR_low[2]]
RR_upp = [-FR_low[0],  FR_upp[1], FR_upp[2]]

RL_low = [-FR_upp[0], -FR_upp[1], FR_low[2]]
RL_upp = [-FR_low[0], -FR_low[1], FR_upp[2]]

ik_low = [FR_low, FL_low, RR_low, RL_low]
ik_upp = [FR_upp, FL_upp, RR_upp, RL_upp]


IK_POS_LOW = np.asarray( [item for sublist in ik_low for item in sublist] )
IK_POS_UPP = np.asarray( [item for sublist in ik_upp for item in sublist] )


""" for IK pybullet computation """
#lower limits for null space
ll = LOWER_ANGLE_JOINT
#upper limits for null space
ul = UPPER_ANGLE_JOINT
#joint ranges for null space
jr = list(np.array(UPPER_ANGLE_JOINT) - np.array(LOWER_ANGLE_JOINT))
#restposes for null space
rp =[0.03, 0.65, -1.2, 
	-0.03, 0.65, -1.2, 
	 0.03, 0.65, -1.2, 
	-0.03, 0.65, -1.2]
#rp = [0]*12
#joint damping coefficents
jd = [0.01] * 12

# make ranges a little larger (easier for computation)
eps = 0.2
ll = [i-eps for i in ll]
ul = [i+eps for i in ul]
jr = [i+eps for i in jr]


""" Control modes, update as more are added """
CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD", "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE" ]

class AlienGo(object):
	""" Pure AlienGo functionalities, NOT a gym interface """
	def __init__(self, 
				pybullet_client,
				isGymInterface=False,
				urdfRoot=URDF_ROOT,
				control_mode = "TORQUE",
				enable_clip_motor_commands=True,
				time_step=0.001,
				actionRepeat=1, 
				self_collision_enabled=True,
				render=False,
				recordVideo=False,
				useFixedBase=False):
		"""Init environment, render should only be true if doing some random tests

		"""
		self._p = pybullet_client
		# to match with google
		#self.pybullet_client = pybullet_client
		self._self_collision_enabled = self_collision_enabled
		self._urdfRoot = urdfRoot
		self._timeStep = time_step
		self._enable_clip_motor_commands = enable_clip_motor_commands
		self._actionRepeat = actionRepeat
		self._isGymInterface = isGymInterface
		self._recordVideo = False
		self.videoLogID = None
		self.render = render
		self._useFixedBase = useFixedBase

		self._step_counter = 0

		if control_mode not in CONTROL_MODES:
			raise ValueError("Must provide a valid control mode!")
		else:
			self._control_mode = control_mode #CONTROL_MODES.index(control_mode)

		# Check if already connected (i.e. if being called from a Gym env)
		connectedStatus = self._p.getConnectionInfo()
		if not connectedStatus['isConnected']:
			if self.render:
				self._p.connect(self._p.GUI)
				self._p.getCameraImage(480,320)
			else:
				self._p.connect(self._p.DIRECT)
		else:
			# We are already connected, and we should render if in GUI mode
			print('\n\n yup \n\n', self._p.GUI, connectedStatus['connectionMethod'], self._p.DIRECT)
			self.render = connectedStatus['connectionMethod']==self._p.GUI

		# motor gains
		self._motor_kps = joint_kp_gains
		self._motor_kds = joint_kd_gains

		self._hard_reset = True
		self.Reset()
		self._hard_reset = False

	##########################################################################
	## Observation related, helpers to get robot states
	## (positions, velocities, contact info, etc.)
	##########################################################################
	def GetSimTime(self):
		""" Get simulation time. """
		return self._step_counter * self._timeStep

	def GetBasePosition(self):
		""" Get the position of Aliengo Base """
		basePos, _ = self._p.getBasePositionAndOrientation(self.quadruped)
		return basePos

	def GetBaseOrientation(self):
		""" Get the base orientation quaternion """
		_, baseOrn = self._p.getBasePositionAndOrientation(self.quadruped)
		return baseOrn

	def GetBaseOrientationRollPitchYaw(self):
		""" Get the base orientation roll pitch yaw """
		baseOrnQuaternion = self.GetBaseOrientation()
		return self._p.getEulerFromQuaternion(baseOrnQuaternion)

	def GetTrueBaseRollPitchYawRate(self):
		"""Get the rate of orientation change of the minitaur's base in euler angle.

		Returns:
			rate of (roll, pitch, yaw) change of the minitaur's base.
		"""
		#angular_velocity = self._p.getBaseVelocity(self.quadruped)[1]
		#orientation = self.GetTrueBaseOrientation()
		angular_velocity = self.GetBaseAngularVelocity()
		orientation = self.GetBaseOrientation()
		return self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)

	def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
		"""Transform the angular velocity from world frame to robot's frame.

		Args:
		angular_velocity: Angular velocity of the robot in world frame.
		orientation: Orientation of the robot represented as a quaternion.

		Returns:
		angular velocity of based on the given orientation.
		"""
		# Treat angular velocity as a position vector, then transform based on the
		# orientation given by dividing (or multiplying with inverse).
		# Get inverse quaternion assuming the vector is at 0,0,0 origin.
		_, orientation_inversed = self._p.invertTransform([0, 0, 0], orientation)
		# Transform the angular_velocity at neutral orientation using a neutral
		# translation and reverse of the given orientation.
		relative_velocity, _ = self._p.multiplyTransforms([0, 0, 0], 
														orientation_inversed, 
														angular_velocity,
														 self._p.getQuaternionFromEuler([0, 0, 0]))
		return np.asarray(relative_velocity)

	def GetBaseOrientationMatrix(self):
		""" Get the base orientation matrix, as numpy array """
		baseOrn = self.GetBaseOrientation()
		return np.asarray(self._p.getMatrixFromQuaternion(baseOrn)).reshape((3,3))

	def GetBaseLinearVelocity(self):
		""" Get base linear velocities (dx, dy, dz) """
		linVel,_ = self._p.getBaseVelocity(self.quadruped)
		return np.asarray(linVel)

	def GetBaseAngularVelocity(self):
		""" Get base angular velocities (droll, dpitch, dyaw) """
		_, angVel = self._p.getBaseVelocity(self.quadruped)
		return np.asarray(angVel)

	def GetJointAngles(self):
		""" Get the joint angle positions (12) """
		jointStates = self._p.getJointStates(self.quadruped,self.jointIds)
		return np.array([state[0] for state in jointStates])

	def GetJointVelocities(self):
		""" Get the joint velocities (12) """
		jointStates = self._p.getJointStates(self.quadruped,self.jointIds)
		return np.array([state[1] for state in jointStates])

	def GetBaseState(self):
		""" Get base state: xyz, rpy, dxyz, drpy  """
		basePos = self.GetBasePosition()
		baseRPY = self.GetBaseOrientationRollPitchYaw()
		baseLinVel = self.GetBaseLinearVelocity()
		baseAngVel = self.GetBaseAngularVelocity()
		return np.concatenate((basePos, baseRPY, baseLinVel, baseAngVel))

	def GetJointState(self):
		""" Get all joint info, (12) pos and (12) vels. """
		joint_pos = self.GetJointAngles()
		joint_vel = self.GetJointVelocities()
		return np.concatenate((joint_pos, joint_vel))

	def GetFullState(self):
		""" Get full robot state:
		-base pos (xyz)
		-base orn (rpy)
		-base linear velocity (dx, dy, dz)
		-base angular velocity (wx, wy, wz)
		-joint pos (12)
		-joint vels (12)
		"""
		return np.concatenate(( self.GetBaseState(), self.GetJointState() ))

	def GetPDGains(self):
		""" Get the current kp and kd values. """
		return self._motor_kps, self._motor_kds

	def SetPDGains(self,kp=None,kd=None):
		""" Set the kp and kd values. """
		curr_kp, curr_kd = self.GetPDGains()
		if kp is not None and len(kp)==len(curr_kp):
			self._motor_kps = kp
		if kd is not None and len(kd)==len(curr_kd):
			self._motor_kds = kd

	def GetAppliedTorques(self):
		""" Get last applied torques.
		If TORQUE control_mode, should be in _last_torques, else can use joint states
		"""
		if self._control_mode == "TORQUE":
			return np.asarray(self._last_torques)
		else:
			joint_states = self._p.getJointStates(self.quadruped,self.jointIds)
			return np.asarray([state[3] for state in joint_states])

	def GetLocalEEPos(self,legID):
		""" Find end effector position of legID in body frame. """
		link_state = self._p.getLinkState(self.quadruped,self.feetIndices[legID],computeLinkVelocity=True, computeForwardKinematics=True)
		# map the world location of the link into local coordinates of the robot body (and then into hip/leg frame)
		basePos, baseOrn = self.GetBasePosition(), self.GetBaseOrientation() 
		linkPos = link_state[4]
		linkOrn = link_state[5]
		invBasePos, invBaseOrn = self._p.invertTransform(basePos,baseOrn)
		linkPosInBase, linkOrnInBase = self._p.multiplyTransforms(invBasePos, invBaseOrn, linkPos, linkOrn)
		linkPosInBase = np.asarray(linkPosInBase)

		# from google (same)
		# base_position, base_orientation = self.GetBasePosition(), self.GetBaseOrientation()
		# inverse_translation, inverse_rotation = self._p.invertTransform(base_position, base_orientation)
		# link_state = self._p.getLinkState(self.quadruped, self.feetIndices[legID])
		# link_position = link_state[0]
		# link_local_position, _ = self._p.multiplyTransforms(inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))
		# print(linkPosInBase, link_local_position)
		return linkPosInBase

	def GetLocalEEVel(self,legID):
		""" Find end effector linear velocity of legID in body frame. """
		print('[DEBUG] This is currently the global e.e. velocity, not local.')
		link_state = self._p.getLinkState(self.quadruped,self.feetIndices[legID],computeLinkVelocity=True, computeForwardKinematics=True)
		# map the world location of the link into local coordinates of the robot body (and then into hip/leg frame)
		basePos, baseOrn = self.GetBasePosition(), self.GetBaseOrientation() 
		baseLinVel, baseAngVel = self.GetBaseLinearVelocity(), self.GetBaseAngularVelocity() 
		linkLinVel = link_state[6]
		linkAngVel = link_state[7]

		baseOrnMatrix = self.GetBaseOrientationMatrix()
		baseLinVelInBodyFrame = baseOrnMatrix.T @ baseLinVel
		linkLinVelInBodyFrame = baseOrnMatrix.T @ linkLinVel
		relative_velocity = linkLinVelInBodyFrame - baseLinVelInBodyFrame

		J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(legID)
		relative_velocity = J @ self.GetJointVelocities()[legID*3:legID*3+3]
		return relative_velocity
		#for v in link_state:
		#	print(v)
		#invBasePos, invBaseOrn = self._p.invertTransform(basePos,baseOrn)
		#linkPosInBase, linkOrnInBase = self._p.multiplyTransforms(invBasePos, invBaseOrn, linkPos, linkOrn)
		#linkPosInBase = np.asarray(linkPosInBase)
		#return np.asarray(linkLinVel), np.asarray(linkAngVel)

	def GetGlobalEEVel(self,legID):
		""" Find global end effector linear and angular velocities of legID. """
		link_state = self._p.getLinkState(self.quadruped,self.feetIndices[legID],computeLinkVelocity=True, computeForwardKinematics=True)
		linkLinVel = link_state[6]
		linkAngVel = link_state[7]
		return np.asarray(linkLinVel), np.asarray(linkAngVel)

	def GetContactInfo(self):
		""" Returns 
		# (1) num valid contacts (only those between a foot and the ground)
		# (2) num invalid contacts (either between a limb and the ground (other than the feet) or any self collisions
		# (3) normal force at each foot (0 if in the air) 
		"""
		# penalize this in reward function ( or terminate condition ? )
		numInvalidContacts = 0
		numValidContacts = 0
		feetNormalForces = [0,0,0,0]

		for c in self._p.getContactPoints():
			# if bodyUniqueIdA is same as bodyUniqueIdB, self collision
			if c[1] == c[2]:
				numInvalidContacts += 1
				continue
			# check link indices, one MUST be a foot index, and the other must be -1 (plane)
			# depending on sim and pybullet version, this can happen in either order (apparently)
			if ( c[3] == -1 and c[4] not in self.feetIndices ) or ( c[4] == -1 and c[3] not in self.feetIndices ):
				numInvalidContacts += 1
			else:
				numValidContacts += 1
				try:
					footIndex = self.feetIndices.index(c[4])
				except:
					footIndex = self.feetIndices.index(c[3])
				feetNormalForces[footIndex] += c[9] # if multiple contact locations?

		return numValidContacts, numInvalidContacts, feetNormalForces

	def IsBodyInContact(self):
		""" Returns bool if base is in contact with the ground or not
		"""
		# penalize this in reward function ( or terminate condition ? )
		#for c in self._p.getContactPoints():
		#	print(c)
		#print(self.plane, self.quadruped)
		#print('here',self._p.getContactPoints(self.plane,self.quadruped,-1,0))
		return self._p.getContactPoints(self.plane,self.quadruped,-1,0) is not ()


	def _terminate(self,fallThreshold=1):
		""" Flag termination if falling down (i.e. body pitch roll too large)
		TODO: add termination for (self) collisions? There seem to be a lot of self collisions as returned by pybullet 
		"""
		curr_pos, curr_ori = self.GetBasePosition(), self.GetBaseOrientationRollPitchYaw()

		# if body is sufficiently rotated, terminate early
		if abs(curr_ori[0]) > fallThreshold or abs(curr_ori[1]) > fallThreshold:
			return True
		else:
			return False

	def DefaultStandUpObs(self,DEBUG_OBS=False):
		""" Default observation for stand up task. Consists of
		- (1 val ) base z
		- (4 vals) base quaternion orn
		- (3 vals) base linear velocities
		- (3 vals) base angular velocities
		- (12 vals) joint angles
		- (12 vals) joint velocities
		- (4 vals) foot normal forces
		Total: 39 values

		Future observations: may want to include relative feet positions, other forces?
		"""
		obs = []

		# get base pos, orientation, linear and angular velocities
		basePos,baseOrn = self.GetBasePosition(), self.GetBaseOrientation()
		linVel, angVel = self.GetBaseLinearVelocity(), self.GetBaseAngularVelocity()
		joint_angles = self.GetJointAngles()
		joint_vels = self.GetJointVelocities()
		_,_, feetNormalForces = self.GetContactInfo()

		obs.extend([basePos[2]])  # only care about z
		obs.extend(baseOrn)
		obs.extend(linVel)
		obs.extend(angVel)
		obs.extend(joint_angles)
		obs.extend(joint_vels)
		#obs.extend(feetNormalForces)

		if DEBUG_OBS:
			print('z', basePos[2] )
			print('baseOrn', baseOrn)
			print('base linVel', linVel)
			print('base angVel', angVel)
			print('joint angles',joint_angles)
			print('joint vels', joint_vels)
			print('normal forces', feetNormalForces)
		
		# Therefore total of 
		# 1 base pos
		# 4 orn
		# 3 lin, 3 ang vels
		# 24 joint pos/vel
		# 4 normal forces
		# = 39 dimensions
		return obs

	def _defaultObsLimits(self):
		""" get max/min observation limits, somewhat conservative """

		z_min = [0]
		quat_low = [-1]*4
		linvel_low = [-5]*3
		angvel_low = [-5]*3
		joint_angles_low = list(LOWER_ANGLE_JOINT)
		joint_vels_low = list(-VELOCITY_LIMITS)
		obs_low = z_min + quat_low + linvel_low + angvel_low + joint_angles_low + joint_vels_low

		z_max = [1]
		quat_upp = [1]*4
		linvel_upp = [5]*3
		angvel_upp = [5]*3
		joint_angles_upp = list(UPPER_ANGLE_JOINT)
		joint_vels_upp = list(VELOCITY_LIMITS)
		obs_upp = z_max + quat_upp + linvel_upp + angvel_upp + joint_angles_upp + joint_vels_upp

		return np.asarray(obs_low), np.asarray(obs_upp)

	##########################################################################
	## Set input torques, PD, etc. anything motor related
	##########################################################################
	def setInputTorques(self,actions):
		""" Set torques for all joints """
		self._last_torques = actions.copy()
		for i,joint in enumerate(self.jointIds):
			self._p.setJointMotorControl2(
						self.quadruped, 
						joint, 
						controlMode=self._p.TORQUE_CONTROL, 
						force=actions[i], 
						maxVelocity=VELOCITY_LIMITS[i])

	def _setMotorTorqueById(self, motor_id, torque):
		""" Set motor torque at motor id """
		self._p.setJointMotorControl2(
			bodyIndex=self.quadruped,
			jointIndex=motor_id,
			controlMode=self._p.TORQUE_CONTROL,
			force=torque)

	def _setMotorTorqueByIds(self, motor_ids, torques):
		""" Set motor torques at motor ids """
		self._p.setJointMotorControlArray(
			bodyIndex=self.quadruped,
			jointIndices=motor_ids,
			controlMode=self._p.TORQUE_CONTROL,
			forces=torques)	

	def setInputPD(self,actions):
		""" Set PD control for all joints """
		for i,joint in enumerate(self.jointIds):
			self._p.setJointMotorControl2(
						self.quadruped, 
						joint, 
						controlMode=self._p.POSITION_CONTROL, 
						targetPosition=actions[i],
						targetVelocity=0, # added 5/26/2020
						positionGain=self._motor_kps[i], #positionGain=10.0,
						velocityGain=self._motor_kds[i], #velocityGain=1, 
						maxVelocity=VELOCITY_LIMITS[i],
						force=TORQUE_LIMITS[i])


	def setPD_joints(self,actions,joints):
		""" Set PD control for certain specified joints (usually called with actions=[0]*4, joints=self.hipIds)"""
		for i,joint in enumerate(joints):
			self._p.setJointMotorControl2(
						self.quadruped, 
						joint, 
						controlMode=self._p.POSITION_CONTROL, 
						targetPosition=actions[i],
						targetVelocity=0, # added 5/26/2020
						positionGain=self._motor_kps[self.jointIds.index(joint)],
						velocityGain=self._motor_kds[self.jointIds.index(joint)], 
						maxVelocity=VELOCITY_LIMITS[self.jointIds.index(joint)],
						force=TORQUE_LIMITS[self.jointIds.index(joint)])

	def setInputIK(self,ee_pos_list):
		""" Called with desired end effector list in order of leg ids, which should be output of _scaleActionsToIK """
		jointPoses = []
		for i in range(4):
			targetPos = ee_pos_list[3*i:3*i+3] 
			jointPoses.extend( self._CalculateIK(i,targetPos) )

		# Clip to joint ranges
		jointPoses = np.clip(jointPoses,LOWER_ANGLE_JOINT,UPPER_ANGLE_JOINT)
		# set joints with PD
		self.setInputPD(jointPoses)

	def setInputForce(self,ee_forces):
		""" Compute and set torque commands based on desired end effector forces, which should be output of _scaleActionsToForces

		As in _scaleActionsToForces,
		If len(actions) == 4,  interpret as four normal forces
		If len(actions) == 12, interpret as four XYZ e.e. desired forces

		Force control (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
		"""
		u = ee_forces.copy()
		if len(u) == 4:
			new_u = np.zeros(12)
			new_u[2::3] = u
			u = new_u
		# compute joint torques (pybullet, vs. i.e. USC software) 
		print(u)
		all_ff_torques_pybullet = np.zeros(12)
		for i in (range(4)):
			linJac, angJac = self._CalculateJacobian(i)
			J = linJac[:,i*3+6:i*3+6+3]
			
			basePos, baseOrn = self.GetBasePosition(), self.GetBaseOrientation() 
			baseOrnMatrix = np.asarray(self._p.getMatrixFromQuaternion(baseOrn)).reshape((3,3))

			# test
			#all_ff_torques_pybullet[i*3:i*3+3] = J.T @ u[i*3:i*3+3]
			all_ff_torques_pybullet[i*3:i*3+3] = np.matmul(J.T , np.matmul( baseOrnMatrix.T , u[i*3:i*3+3]) )

		# clip to torque limits
		#print('u before directions', all_ff_torques_pybullet)
		#all_ff_torques_pybullet = all_ff_torques_pybullet * JOINT_DIRECTIONS
		u = np.clip(all_ff_torques_pybullet,-TORQUE_LIMITS,TORQUE_LIMITS)
		#print('u after directions', u)
		# set torques 
		self.setInputTorques(u)
		#self.setPD_joints([0]*4,self.hipIds)


	def _disable_motor(self):
		""" Disable motor (completely loosen), to be used in Reset(), because previous torques were being carried over """
		for j in range(self._p.getNumJoints(self.quadruped)):
			self._p.setJointMotorControl2(self.quadruped,
								  j,
								  controlMode=self._p.POSITION_CONTROL,
								  targetPosition=0,
								  targetVelocity=0,
								  positionGain=0.1,
								  velocityGain=0.1,
								  force=0)
		
	##########################################################################
	## Scale actions from [-1, 1] to ranges for torque control, PD, IK, etc. 
	##########################################################################
	def _scaleActionsToTorques(self,actions):
		""" Scale actions in [-1,1] range to torque limits, clip based on torque limits """
		u = TORQUE_LIMITS * np.clip(actions,-1,1)
		return u

	def _scaleActionsToTorquesNoHips(self,actions):
		""" Scale actions in [-1,1] range to torque limits, clip based on torque limits 
		Assuming ony thigh and calf values, no hips.
		"""
		u = TORQUE_LIMITS_NO_HIPS * np.clip(actions,-1,1)
		full_u = [0, u[0], u[1], 0, u[2], u[3],
				  0, u[4], u[5], 0, u[6], u[7]]
		return np.asarray(full_u)

	def _scaleActionsToPositions(self,actions):
		""" Scale actions in [-1,1] range to position limits, clip based on joint limits """
		# Scaled u based on the following:
		# b is new upper limit,   a is new lower limit
		# max is old upper limit, max is old lower limit
		#        (b-a)(x - min)
		# f(x) = --------------  + a
		#           max - min
		# u = ( ( UPPER_ANGLE_JOINT - LOWER_ANGLE_JOINT ) * (u - (-1) ) ) / (1 - (-1)) + LOWER_ANGLE_JOINT
		# u = LOWER_ANGLE_JOINT +  0.5 * ( UPPER_ANGLE_JOINT - LOWER_ANGLE_JOINT ) * (u + 1) 

		u = np.clip(actions,-1,1)
		u = LOWER_ANGLE_JOINT +  0.5 * ( UPPER_ANGLE_JOINT - LOWER_ANGLE_JOINT ) * (u + 1) 
		u = np.clip(u,LOWER_ANGLE_JOINT,UPPER_ANGLE_JOINT)
		#nicePrint(u)
		return u

	def _scaleActionsToPositionsNoHips(self,actions):
		""" Scale actions in [-1,1] range to position limits, clip based on joint limits 
		Assuming ony thigh and calf values, no hips.
		"""
		u = np.clip(actions,-1,1)
		u = LOWER_ANGLE_JOINT_NO_HIPS +  0.5 * ( UPPER_ANGLE_JOINT_NO_HIPS - LOWER_ANGLE_JOINT_NO_HIPS ) * (u + 1) 
		u = np.clip(u,LOWER_ANGLE_JOINT_NO_HIPS,UPPER_ANGLE_JOINT_NO_HIPS)
		full_u = [0, u[0], u[1], 0, u[2], u[3],
				  0, u[4], u[5], 0, u[6], u[7]]
		return np.asarray(full_u)

	def _scaleActionsToIK(self,actions):
		""" Scale actions in [-1,1] range to corresponding desired x,y,z positions """
		u = np.clip(actions,-1,1)
		targetPositions = IK_LOW +  0.5 * ( IK_UPP - IK_LOW ) * (u + 1) 
		return targetPositions

	def _scaleActionsToForces(self,actions):
		""" Scale actions in [-1,1] range to corresponding desired normal forces, to torque commands

		If len(actions) == 4,  interpret as four normal forces
		If len(actions) == 12, interpret as four XYZ e.e. desired forces

		Force control (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
		"""
		u = np.clip(actions,-1,1)
		if len(u) == 4:
			# Fz only
			u = MIN_NORMAL_FORCE + 0.5 * (MAX_NORMAL_FORCE - MIN_NORMAL_FORCE) * (u+1)
			u = np.clip(u,MIN_NORMAL_FORCE,MAX_NORMAL_FORCE)
			u_norms_desired = u.copy()
			# copy 0 for desired Fx and Fy 
			new_u = np.zeros(12)
			new_u[2::3] = u
			u = new_u
		else:
			# Fx, Fy, Fz at each foot
			u = MIN_XYZ_FORCE + 0.5 * (MAX_XYZ_FORCE - MIN_XYZ_FORCE) * (u+1)
			u = np.clip(u,MIN_XYZ_FORCE,MAX_XYZ_FORCE)
			u_norms_desired = u.copy()[2::3]

		return u, u_norms_desired


	##########################################################################
	## Helper functions for IK, Jacobians, etc. 
	##########################################################################
	def _CalculateIK(self,leg_id,targetPos):
		""" Calculate IK for legid, with local target position.

		TODO: verify how many iterations of IK needed, as well as residual threshold.
		See Aliengo/pybullet_tests/aliengo_ik_tests.py
		"""
		# need current joint states
		currJointStates = self._p.getJointStates(self.quadruped,self.jointIds)
		currJointPos = [i[0] for i in currJointStates]
		ee_index = self.feetIndices[leg_id]
		jointPosesCurr = self._p.calculateInverseKinematics(self.quadruped, 
												ee_index, 
												targetPos, 
												lowerLimits=ll, 
												upperLimits=ul,
												jointRanges=jr, 
												restPoses=rp,
												solver=0,
												currentPositions=currJointPos,
												residualThreshold=.0001,
												maxNumIterations=200)

		#print(jointPosesCurr)
		return jointPosesCurr[3*leg_id:3*leg_id+3]

	def _CalculateJacobian(self,leg_id):
		""" Calculate Jacobian for leg_id end effector """
		# need current joint states
		currJointStates = self._p.getJointStates(self.quadruped,self.jointIds)
		currJointPos = [i[0] for i in currJointStates]
		currJointVel = [i[1] for i in currJointStates]

		linJac, angJac = self._p.calculateJacobian(self.quadruped, 
							self.feetIndices[leg_id], 
							[0,0,0],
							currJointPos,
							[0]*12, #currJointVel, 
							[0]*12)

		#return linJac, angJac
		return np.asarray(linJac), np.asarray(angJac)


	def _ComputeJacobianAndPositionUSC(self, legID):
		"""Translated to python/pybullet from: https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/LegController.cpp
		can also use pybullet function calculateJacobian()
		IN LEG FRAME
		(see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
		# from LegController:
		# * Leg 0: FR; Leg 1: FL;
		# * Leg 2: RR ; Leg 3: RL;

		# hip locations in body frame
		# FR 0.2399 -0.051
		# FL 0.2399 -0.051
		# RR -0.2399 -0.051
		# RL -0.2399 0.051
		"""

		# joint positions of leg
		q = self.GetJointAngles()[legID*3:legID*3+3]
		# Foot position in base frame
		linkPosInBase = self.GetLocalEEPos(legID)
		# local position of E.E. in leg frame (relative to hip)
		#pos = np.array(linkPosInBase) - HIP_POSITIONS[legID]
		pos = np.zeros(3)

		l1 = 0.083
		l2 = 0.25
		l3 = 0.25

		sideSign = 1
		if legID == 0 or legID == 2:
			sideSign = -1

		s1 = np.sin(q[0])
		s2 = np.sin(q[1])
		s3 = np.sin(q[2])

		c1 = np.cos(q[0])
		c2 = np.cos(q[1])
		c3 = np.cos(q[2])

		c23 = c2 * c3 - s2 * s3
		s23 = s2 * c3 + c2 * s3

		J = np.zeros((3,3))

		J[1, 0] = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
		J[2, 0] =  sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
		J[0, 1] = -l3 * c23 - l2 * c2
		J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
		J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
		J[0, 2] = -l3 * c23
		J[1, 2] = -l3 * s23 *s1
		J[2, 2] = l3 * s23 * c1  

		# E.E. pos
		pos[0] = -l3 * s23 - l2 * s2
		pos[1] = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
		pos[2] = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

		return J, pos

	##########################################################################
	## Internal Step function, which calls Apply/clip action based on mode
	##########################################################################
	def _ClipActions(self, actions, control_mode=None):
		""" Clip and/or scale actions according to control_mode """
		if control_mode is None:
			#default to whatever is current
			control_mode = self._control_mode

		if control_mode == "TORQUE":
			u = self._scaleActionsToTorques(actions)
		elif control_mode == "TORQUE_NO_HIPS":
			u = self._scaleActionsToTorquesNoHips(actions)
		elif control_mode == "PD":
			u = self._scaleActionsToPositions(actions)
		elif control_mode == "PD_NO_HIPS":
			u = self._scaleActionsToPositionsNoHips(actions)
		elif control_mode == "IK":
			u = self._scaleActionsToIK(actions)
		elif control_mode == "FORCE":
			u,_ = self._scaleActionsToForces(actions)
		else:
			"""TODO: add impedance control 
			"""
			raise ValueError("This control mode not implemented yet.")

		# if enable_clip_motor_commands, then clip additionally ?
		return u

	def ApplyAction(self, motor_commands, control_mode=None):
		""" Set motor commands according to control mode.
		Assumes sending valid commands, i.e. called _ClipActions first if using from Gym env
		"""
		if control_mode is None:
			#default to whatever is current
			control_mode = self._control_mode

		if control_mode == "TORQUE":
			self.setInputTorques(motor_commands)
		elif control_mode == "TORQUE_NO_HIPS":
			self.setInputTorques(motor_commands)
			# keep hips at 0
			self.setPD_joints([0]*4,self.hipIds)
		elif control_mode == "PD":
			self.setInputPD(motor_commands)
			# keep hips at 0
			self.setPD_joints([0]*4,self.hipIds)
		elif control_mode == "PD_NO_HIPS":
			self.setInputPD(motor_commands)
		elif control_mode == "IK":
			self.setInputIK(motor_commands)
		elif control_mode == "FORCE":
			self.setInputForce(motor_commands)
		else:
			"""TODO: add impedance control 
			"""
			raise ValueError("This control mode not implemented yet.")


	def Step(self,u):
		""" Step simulation. Reward should be calculated ELSEWHERE SINCE THIS IS NOT A GYM ENV """
		#assert len(u) == len(self.action_space.low)
		self._last_base_pos = self.GetBasePosition()
		self._last_base_vel = self.GetBaseLinearVelocity()
		self._last_base_orn = self.GetBaseOrientationRollPitchYaw()

		# if using as gym, will need to clip the actions
		if self._isGymInterface:
			proc_action = self._ClipActions(u)
		else:
			proc_action = u

		#print('action', proc_action)
		# TODO: process action, i.e. filters, noise, motor models, etc.
		for i in range(self._actionRepeat):
			
			self.ApplyAction(proc_action)
			self._p.stepSimulation()
			self._step_counter += 1

			if self.render:
				time.sleep(max(self._timeStep,0.01))

		return

	##############################################################################
	## Setup quadruped, reset, add noise or perturb states, etc.
	##############################################################################
	def _setUpQuadruped(self):
		"""Reset entire environment, load quadruped. Should be called for any hard reset.
		"""
		self._p.resetSimulation()
		self._p.setTimeStep(self._timeStep)
		self._p.setGravity(0,0,-9.8)
		self._p.setPhysicsEngineParameter(enableConeFriction=0)
		self.plane = self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))

		if self._self_collision_enabled:
			self.quadruped = self._p.loadURDF(os.path.join(self._urdfRoot,URDF_FILENAME), 
											[0,0,0.48],[0,0,0,1],flags=self._p.URDF_USE_SELF_COLLISION, useFixedBase=self._useFixedBase)
		else:
			self.quadruped = self._p.loadURDF(os.path.join(self._urdfRoot,URDF_FILENAME), 
											[0,0,0.48],[0,0,0,1], useFixedBase=self._useFixedBase)

		# These are the joints of interest: Hip, Thigh and Calf joints for each of the legs, so total number of ID's are 12
		num_joints = self._p.getNumJoints(self.quadruped)

		self.jointIds=[]
		self.feetIndices = []
		self.hipIds = []
		self.thighIds = []
		self.calfIds = []
		for j in range (num_joints):
			info = self._p.getJointInfo(self.quadruped,j)
			# Remove default joint damping
			#self._p.changeDynamics(self.quadruped,j,linearDamping=0, angularDamping=0)
			self._p.changeDynamics(info[0],-1,linearDamping=0, angularDamping=0)
			#info = self._p.getJointInfo(self.quadruped,j)

			joint_name = info[1].decode("UTF-8")
			if _HIP_NAME_PATTERN.match(joint_name):
				self.hipIds.append(j)
			elif _THIGH_NAME_PATTERN.match(joint_name):
				self.thighIds.append(j)
			elif _CALF_NAME_PATTERN.match(joint_name):
				self.calfIds.append(j)
			elif _FOOT_NAME_PATTERN.match(joint_name):
				self.feetIndices.append(j)
			jointType = info[2]
			if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
				self.jointIds.append(j)
				# apparently this loosens the joint, please check exact use of this line
				self._p.setJointMotorControl2(self.quadruped, j, controlMode=p.VELOCITY_CONTROL, targetVelocity=0,force=0)

		# sort, just to make sure IDs are in the right order
		self.jointIds.sort()
		self.feetIndices.sort()
		self.hipIds.sort()
		self.thighIds.sort()
		self.calfIds.sort()

		# find total mass:
		self.total_mass = 0
		# add base mass as well! it's index is -1
		self.total_mass += self._p.getDynamicsInfo(self.quadruped,-1)[0]
		for j in range (self._p.getNumJoints(self.quadruped)):
			linkDyn = self._p.getDynamicsInfo(self.quadruped,j)
			self.total_mass += linkDyn[0]

		# # loosen joints (removes damping)
		# for j in range(num_joints):
		# 	self._p.setJointMotorControl2(self.quadruped, j, controlMode=p.VELOCITY_CONTROL, targetVelocity=0,force=0)

	def SetPoseFromHeight(self, baseHeight=0.5 ):
		""" Take in desired base position, and compute the ee xyz positions to be above ground. """
		nominal_foot_offset = 0.03
		hip_pos =  HIP_POSITIONS_RESET.copy()
		desFeetPos = [ pos + np.array([0,0,-baseHeight + nominal_foot_offset]) for pos in hip_pos]
		#desFeetPos[2::3] = [-baseHeight + nominal_foot_offset] * 4

		jointPoses = []
		for i in range(4):
			targetPos = desFeetPos[i]
			jointPosesCurr = self._CalculateIK(i,targetPos)
			jointPoses.extend(jointPosesCurr)#3*i:3*i+3]

		# reset joints
		for i, jointPos in enumerate(jointPoses):
			p.resetJointState(self.quadruped, self.jointIds[i], jointPos, targetVelocity=0)

		# verify solution
		for i in range(4):
			# verify solution 
			localEEpos = self.GetLocalEEPos(i)
			print('desired', desFeetPos[i], 'actual', localEEpos)
		print("joints", jointPoses)

		self._p.resetBasePositionAndOrientation(self.quadruped,[0,0,baseHeight],[0,0,0,1])

	def SetJointStates(self,joint_states):
		"""Take in joint positions and velocities, and set joint states. """
		joint_pos, joint_vel = np.split(joint_states, 2)
		for i,jointId in enumerate(self.jointIds):
			self._p.resetJointState(self.quadruped, jointId, joint_pos[i], targetVelocity=joint_vel[i])
		return joint_pos, joint_vel

	def SetPoseAndSettle(self,basePosOrn,joint_states, settle_threshold=0.01):
		""" Set robot base pos, joint states, and simulate until velocities are under some threshold. """
		base_pos, base_orn = np.split(basePosOrn, 2)
		self.ResetBasePositionAndOrientation(base_pos,base_orn)
		joint_pos, joint_vel = self.SetJointStates(joint_states)
		self.setInputPD(joint_pos)
		self._settle(settle_threshold)

		# make sure the base position and orientation is close to what we wanted!
		actual_basePosOrn = self.GetBaseState()[0:6]
		#print(basePosOrn,actual_basePosOrn)
		assert np.linalg.norm(basePosOrn - actual_basePosOrn) < 0.1
		#self._disable_motor()

	def _settle(self,settle_threshold=0.2, max_settle_sim_time=5):
		""" Simulate until robot overall velocity falls below a threshold. Call after setting pose. """
		settle_sim_cntr = 0
		while True:
			self._p.stepSimulation()
			all_vels = [self.GetBaseLinearVelocity(), self.GetBaseAngularVelocity(), self.GetJointVelocities()]
			total_vel_scalar = np.sum( [np.linalg.norm(i) for i in all_vels] )
			if total_vel_scalar < settle_threshold:
				break
			if settle_sim_cntr * self._timeStep > max_settle_sim_time:
				print('[DEBUG] Simulated' , max_settle_sim_time , 'and still going.')
				break
			settle_sim_cntr += 1
			if self.render:
				time.sleep(0.01)

	def SetFullState(self,bodyPos,bodyOrn,bodyLinVel=None,bodyAngVel=None,jointPositions=None,jointVels=None):
		""" Set robot state. (Usually) to visualize state. """
		self.ResetBasePositionAndOrientation(bodyPos,bodyOrn)
		if bodyLinVel is not None and bodyAngVel is not None:
			self.ResetBaseVelocity(bodyLinVel,bodyAngVel)
		elif bodyLinVel is not None and bodyAngVel is None:
			self.ResetBaseVelocity(bodyLinVel,[0,0,0])
		if jointVels is None:
			jointVels=np.zeros(len(jointPositions))
		
		for i,jointPos in enumerate(jointPositions):
			self._p.resetJointState(self.quadruped, self.jointIds[i], jointPos, targetVelocity=jointVels[i])

		if self.render:
			#self._p.stepSimulation()
			time.sleep(max(self._timeStep,0.01))

	def ResetBasePositionAndOrientation(self, base_pos, base_orn ):
		""" Resets base position and orientation. Orientation can be Euler or quaternion """
		if len(base_orn) == 3: #Euler, but need quaternion
			base_orn = self._p.getQuaternionFromEuler(base_orn)
		self._p.resetBasePositionAndOrientation(self.quadruped,base_pos,base_orn)

	def ResetBaseVelocity(self, base_linvel, base_angvel ):
		""" Resets base linear and angular velocities. """
		self._p.resetBaseVelocity(self.quadruped,base_linvel, base_angvel)

	def FormatFRtoAll(self,FR_pos):
		""" Format position in FR frame to all other limbs.
		Takes in FR_pos as numpy array, returns list of numpy arrays for the same FR_pos in each frame. """
		FL_pos = np.array( [ FR_pos[0], -FR_pos[1], FR_pos[2]])
		RR_pos = np.array( [-FR_pos[0],  FR_pos[1], FR_pos[2]])
		RL_pos = np.array( [-FR_pos[0], -FR_pos[1], FR_pos[2]])
		return [ FR_pos, FL_pos, RR_pos, RL_pos ]
		
	def Reset(self,state="Stand",hardReset=False,addNoise=False):
		""" Reset AlienGo to initial states.
		Can specify state in {"Stand", "Sit", "RandValid", "Zeros"}
		"""
		# load whole quadruped again, avoid since this is slow/unnecessary (usually)
		if self._hard_reset or hardReset:
			self._setUpQuadruped()
		# problem with reset - it will keep the last applied motor torques, disable
		self._disable_motor()
		self._step_counter = 0
		# default initial position
		init_position = [0.0, 1.2217304764, -2.77507351067,
						 0.0, 1.2217304764, -2.77507351067,
						 0.0, 1.2217304764, -2.77507351067,
						 0.0, 1.2217304764, -2.77507351067]
		if addNoise:
			# add some noise in the joints
			init_position = np.asarray(init_position) + .5 * (np.random.random(12)-0.5)

		# All zero joint angles, just dangle
		if state == "Zeros":
			start_baseHeight = 0.5
			init_position = [0]*12
			for i, joint_pos in enumerate(init_position):
				p.resetJointState(self.quadruped, self.jointIds[i], joint_pos, targetVelocity=0)

		# Sit (legs drawn in, torso close to ground)
		if state == "Sit":
			start_baseHeight = 0.25
			sit_init_position = init_position
			for i, sit_pos in enumerate(sit_init_position):
				self._p.resetJointState(self.quadruped, self.jointIds[i], sit_pos, targetVelocity=0)
			init_position = sit_init_position
		
		# Stand (legs extended, check what makes sense?)		
		if state == "Stand":
			start_baseHeight = 0.45
			stand_init_position = [  0.03, 0.65, -1.2,
									-0.03, 0.65, -1.2,
									 0.03, 0.65, -1.2,
									-0.03, 0.65, -1.2] 
			# stand_init_position = [0.19404342450218479, 0.5675874782033202, -1.1348414237915125, 
			# 						-0.19404342450218479, 0.5675874782033202, -1.1348414237915125, 
			# 						0.1940434245021848, 0.5675874782033202, -1.1348414237915123, 
			# 						-0.1940434245021848, 0.5675874782033202, -1.1348414237915123]
			for i, stand_pos in enumerate(stand_init_position):
				self._p.resetJointState(self.quadruped, self.jointIds[i], stand_pos, targetVelocity=0)
			init_position = stand_init_position

		if addNoise:
			# random height
			start_baseHeight = max(0.45*np.random.random(),0.2)
		else:
			"""TODO: verify this is a good starting height
			"""
			if self._useFixedBase:
				start_baseHeight = 1.0
		self._p.resetBasePositionAndOrientation(self.quadruped,[0,0,start_baseHeight],[0,0,0,1])

		#Make sure legs don't start IN ground.. so should reset base first, and then choose valid leg configuration based on IK
		if state == "RandValid":
			# choose valid initial conditions for legs based on valid IK from current robot height
			basePos = self.GetBasePosition()
			nominal_foot_offset = 0.05
			# might want to change after adding in orientation noise
			z_FR = np.array([0,0, -basePos[2] + nominal_foot_offset])
			z_FL = np.array([0,0, -basePos[2] + nominal_foot_offset])
			z_RR = np.array([0,0, -basePos[2] + nominal_foot_offset])
			z_RL = np.array([0,0, -basePos[2] + nominal_foot_offset])

			# hip locations
			# FR_hip = np.array([ 0.2399 , -0.051, 0])
			# FL_hip = np.array([ 0.2399 ,  0.051, 0])
			# RR_hip = np.array([-0.2399 , -0.051, 0])
			# RL_hip = np.array([-0.2399 ,  0.051, 0])
			# for better stability 
			FR_hip_reset = np.array([ 0.2399 , -0.11, 0])
			FL_hip_reset = np.array([ 0.2399 ,  0.11, 0])
			RR_hip_reset = np.array([-0.2399 , -0.11, 0])
			RL_hip_reset = np.array([-0.2399 ,  0.11, 0])

			desiredFootLocations = [ FR_hip_reset + z_FR,
									 FL_hip_reset + z_FL,
									 RR_hip_reset + z_RR,
									 RL_hip_reset + z_RL]

			jointPoses = []
			for i in range(4):
				targetPos = desiredFootLocations[i]
				jointPosesCurr = self._CalculateIK(i,targetPos)
				jointPoses.extend(jointPosesCurr)
			# hips should be 0 for better stability
			jointPoses[0::3] = [0]*4
			# reset joints
			for i, jointPos in enumerate(jointPoses):
				p.resetJointState(self.quadruped, self.jointIds[i], jointPos, targetVelocity=0)

			# verify solution
			for i in range(4):
				# verify solution 
				localEEpos = self.GetLocalEEPos(i)
				#print('desired', desiredFootLocations[i], 'actual', localEEpos)

		# vary number of initial simulations, simulate until get contact
		beg_sims = round(20*np.random.random())
		for _ in range(beg_sims):
			#self.setPD(init_position)
			#self.setTorques([0]*12)
			# if self._p.getContactPoints() != ():
			# 	# break if already have a contact
			# 	break
			self._p.stepSimulation()
			if self.render:
				time.sleep(1./500.)
		# disable motor depending on mode
		self._disable_motor()

		if self.render:
			self.configureVisualizer()
			if self._recordVideo:
				self.RecordVideoHelper()
		
		return


	def RandomInitNoise(self):
		""" initialize robot into random configuration. TODO: make sure no limb is IN the ground. 
		Check valid limits.
		"""
		# These are the current rand limits
		x_limit = y_limit = 0.5
		z_upp = 0.5
		z_low = 0.3
		roll_limit = 0.5
		pitch_limit = 0.5
		yaw_limit = 2*np.pi
		# velocity limits
		linvel_limit = 0.1
		roll_rate_limit = 0.1
		pitch_rate_limit = 0.1
		yaw_rate_limit = 0.2

		# Set base x,y,z
		dist = x_limit * 2 * (0.5 - np.random.random())
		ang = 2.*np.pi*np.random.random()
		x = dist*np.sin(ang)
		y = dist*np.sin(ang)
		z = max(z_upp*np.random.random(),z_low)
		# Set roll, pitch, yaw
		roll = roll_limit * 2 * (0.5 - np.random.random())
		pitch = pitch_limit * 2 * (0.5 - np.random.random())
		yaw = yaw_limit * 2 * (0.5 - np.random.random())
		# set linear velocities
		vx, vy, vz = linvel_limit * 2 * (0.5 - np.random.random(3))
		# set angular velocities
		wx = roll_rate_limit  * 2 * (0.5 - np.random.random())
		wy = pitch_rate_limit * 2 * (0.5 - np.random.random())
		wz = yaw_rate_limit   * 2 * (0.5 - np.random.random())

		baseOrnQuaternion = self._p.getQuaternionFromEuler([roll,pitch,yaw])
		self._p.resetBasePositionAndOrientation(self.quadruped, [x,y,z], baseOrnQuaternion)
		self._p.resetBaseVelocity(self.quadruped,[vx, vy, vz], [wx, wy, wz])

		# For each joint, set random state
		self._perturbJointStates()

	def _perturbJointStates(self,posNoise=0.2,velNoise=0.2):
		""" add some noise to current configuration """
		jointPos = np.asarray(self.GetJointAngles())
		jointVels= np.asarray(self.GetJointVelocities())

		jointPos = jointPos + posNoise * 2 *(0.5-np.random.random(len(self.jointIds)))
		jointVels= jointVels+ velNoise * 2 *(0.5-np.random.random(len(self.jointIds)))

		# clip limits
		jointPos = np.clip(jointPos,LOWER_ANGLE_JOINT,UPPER_ANGLE_JOINT)
		jointVels= np.clip(jointVels,-VELOCITY_LIMITS,VELOCITY_LIMITS)

		# reset joint states
		for i,jointId in enumerate(self.jointIds):
			self._p.resetJointState(self.quadruped, jointId, jointPos[i], targetVelocity=jointVels[i])

	##########################################################################
	## Helper functions for I/O, videos, visualizer, etc.
	##########################################################################
	def startRecordingVideo(self,name):
		self.videoLogID = self._p.startStateLogging(self._p.STATE_LOGGING_VIDEO_MP4, name)

	def stopRecordingVideo(self):
		self._p.stopStateLogging(self.videoLogID)

	def RecordVideoHelper(self):
		""" Helper to record video, if not already, or end and start a new one """
		# If no ID, this is the first video, so make a directory and start logging
		if self.videoLogID == None:
			directoryName = VIDEO_LOG_DIRECTORY + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")
			assert isinstance(directoryName, str)
			os.makedirs(directoryName, exist_ok=True)
			self.videoDirectory = directoryName

		else:
			# stop recording and record a new one
			self.stopRecordingVideo()

		output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"
		logID = self.startRecordingVideo(output_video_filename)
		self.videoLogID = logID

	def close(self):
		if self._recordVideo:
			self.stopRecordingVideo()
		self._p.disconnect()

	def configureVisualizer(self):
		""" Remove all visualizer borders, and zoom in """
		# get rid of random visualizer things
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
		self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI,0)
		# set camera
		vis_cam = self._p.getDebugVisualizerCamera()
		pos, orn = self.GetBasePosition(), self.GetBaseOrientation()
		self._p.resetDebugVisualizerCamera(vis_cam[10], vis_cam[8], vis_cam[9], pos)
		self._p.resetDebugVisualizerCamera(1.5, vis_cam[8], vis_cam[9], [pos[0],pos[1],pos[2]])

if __name__ == "__main__":
	# test out some functionalities
	control_mode = "PD"
	robot=AlienGo(pybullet_client=p,control_mode=control_mode,render=True ,useFixedBase=False)
	print(robot.total_mass)
	#sys.exit()
	while True:
		if control_mode == "TORQUE":
			u = [10,0,0]*4
		elif control_mode=="FORCE":
			u = [robot.total_mass * (-9.8/4)] * 4
			#u = [-56]*4
		elif control_mode=="PD":
			u = INIT_STAND_POSITION
			u = [ 0.000,  1.257, -2.356,  0.000,  1.257, -2.356,  0.000,  1.257, -2.356,  0.000,  1.257, -2.356 ]
			u = [0]*12
		else:
			u = [0]*12
		robot.Step(u)
		_,_,normals = robot.GetContactInfo()
		print('actual normals', sum(normals), 'expected', robot.total_mass*9.8)
		linJac, angJac = robot._CalculateJacobian(1)
		J,pos = robot._ComputeJacobianAndPositionUSC(1)
		#nicePrint2D(linJac)
		#nicePrint2D(J)
		#robot.GetLocalEEPos(0)