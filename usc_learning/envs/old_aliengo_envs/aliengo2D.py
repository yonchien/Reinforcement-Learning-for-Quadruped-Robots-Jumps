""" This file extends the AlienGo robot in 2D in pybullet. 
	-Note: this is not intended to be used as Gym environment on its own, but should be called from RL or IL env.

	Organization:
	-See aliengo.py 
	-The base link is now different, hence a lot of the function overriding.
	-Actions are mirrored across x axis, so hips should be kept at 0
	-Mainly care about [Front_thigh, Front_calf, Rear_thigh, Rear_calf]

	inspired from the minitaur/laikago pybullet classes
"""
import time, datetime
import numpy as np
from gym import spaces
import sys, os
import re

# remove importing pybullet afterwards? just for testing
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bullet_client

from usc_learning.envs import aliengo 
from usc_learning.envs.aliengo import _HIP_NAME_PATTERN, _THIGH_NAME_PATTERN, _CALF_NAME_PATTERN, _FOOT_NAME_PATTERN
from usc_learning.envs.aliengo import UPPER_ANGLE_JOINT, LOWER_ANGLE_JOINT, TORQUE_LIMITS, VELOCITY_LIMITS
from usc_learning.utils.utils import nicePrint, nicePrint2D

URDF_ROOT = os.path.join(os.path.dirname(__file__), 'pyb_data/')
""" URDF and strings """
#URDF_ROOT = './pyb_data/' #pybullet_data.getDataPath()
#URDF_FILENAME = "aliengo/urdf/aliengo2D.urdf"
URDF_FILENAME = "aliengo/urdf/aliengo2D_constraintsAtEnd_v2.urdf"
VIDEO_LOG_DIRECTORY = 'videos/' 

#BASE_STRING = b'floating_base'
_BASE_STRING = re.compile(r"\w*floating_base\w*")

""" Joint position, velocity, and torque limits, and PD gains """
# thigh joints have no built in limit (what makes sense?)
UPPER_ANGLE_JOINT_2D = [3.14, -0.645771823238 ,
					    3.14, -0.645771823238 ]
LOWER_ANGLE_JOINT_2D = [-3.14, -2.77507351067 ,
					    -3.14, -2.77507351067]


UPPER_ANGLE_JOINT_2D = np.array(UPPER_ANGLE_JOINT_2D)
LOWER_ANGLE_JOINT_2D = np.array(LOWER_ANGLE_JOINT_2D)

TORQUE_LIMITS_2D = np.asarray( [44,55] * 2 )
VELOCITY_LIMITS_2D = np.asarray( [20,16] * 2 )

# right thigh/calf ids
RIGHT_SIDE_IDS = np.array([1,2, 7,8])

# from https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/body.cpp
""" TODO: the below gains need to be tuned 
"""
joint_kp_gains = [70,180,300]*4
joint_kd_gains = [3,8,15] * 4

# joint_kp_gains = [1]*12
# joint_kd_gains = [.2] * 12

joint_kp_gains = [1]*8 + [10]*4
joint_kd_gains = [.2]*8 + [1]*4

joint_kp_gains = [70,180,300]*4
joint_kd_gains = [3,8,15] * 4

joint_kp_gains = [300,300,300]*4
joint_kd_gains = [20,20,20] * 4

# joint_kp_gains = [70,180,300]*4
# joint_kd_gains = [3,8,15] * 4

""" Current force control environment limits  """
NORMAL_FORCE_HIGH = 80
MIN_NORMAL_FORCE = np.asarray([-NORMAL_FORCE_HIGH]*2)
MAX_NORMAL_FORCE = np.asarray([-30]*2)
#MIN_XYZ_FORCE = np.asarray( [-20,-20,-60]*4)
#MAX_XYZ_FORCE = np.asarray( [ 20, 20,-40]*4)
MIN_XZ_FORCE = np.asarray( [-20,-NORMAL_FORCE_HIGH]*2)
MAX_XZ_FORCE = np.asarray( [ 20,-30]*2)

""" Control modes, update as more are added """
CONTROL_MODES = [ "TORQUE", "PD", "IK", "FORCE", "IMPEDANCE" ]

class AlienGo2D(aliengo.AlienGo):
	""" Pure AlienGo2D functionalities, NOT a gym interface """
	def __init__(self, 
				pybullet_client,
				isGymInterface=False,
				urdfRoot=URDF_ROOT,
				control_mode = "TORQUE",
				render=False,
				useFixedBase=False,
				time_step=0.001,
				actionRepeat=1):

		super(AlienGo2D, self).__init__(pybullet_client=pybullet_client,
										isGymInterface=isGymInterface,
										urdfRoot=urdfRoot,
										control_mode = control_mode,
										render=render,
										useFixedBase=useFixedBase,
										time_step=time_step,
										actionRepeat=actionRepeat)

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
	def _getBasePositionAndOrientation2D(self):
		""" Internal helper to go get true base pos and orn. """
		base_state = self._p.getLinkState(self.quadruped,self.real_base,computeLinkVelocity=True, computeForwardKinematics=True)
		basePos = base_state[4]
		baseOrn =  base_state[5]
		return basePos, baseOrn

	def _getBaseLinearAndAngularVelocity2D(self):
		""" Internal helper to go get true base velocities. """
		base_state = self._p.getLinkState(self.quadruped,self.real_base,computeLinkVelocity=True, computeForwardKinematics=True)
		baseLinVel = base_state[6]
		baseAngVel =  base_state[7]
		return baseLinVel, baseAngVel

	def GetBasePosition(self):
		""" Get the position of Aliengo Base. Now using real_base instead of getBasePosandOrn since new base """
		basePos, _ = self._getBasePositionAndOrientation2D()
		return basePos

	def GetBaseOrientation(self):
		""" Get the base orientation quaternion """
		_, baseOrn = self._getBasePositionAndOrientation2D()
		return baseOrn

	def GetBaseLinearVelocity(self):
		""" Get base linear velocities (dx, dy, dz) """
		linVel,_ = self._getBaseLinearAndAngularVelocity2D()
		return np.asarray(linVel)

	def GetBaseAngularVelocity(self):
		""" Get base angular velocities (droll, dpitch, dyaw) """
		_, angVel = self._getBaseLinearAndAngularVelocity2D()
		return np.asarray(angVel)

	def GetBaseState2D(self):
		""" Get base state: x,z,phi, dx,dz,dphi """
		basePos = self.GetBasePosition()
		baseRPY = self.GetBaseOrientationRollPitchYaw()
		baseLinVel = self.GetBaseLinearVelocity()
		baseAngVel = self.GetBaseAngularVelocity()
		return np.asarray([ basePos[0], basePos[2], baseRPY[1], baseLinVel[0], baseLinVel[2], baseAngVel[1] ])

	def GetJointState2D(self):
		""" Get the right side indices. [qFront, qRear, dqFront, dqRear] (8 vals)"""
		joint_pos = self.GetJointAngles()[RIGHT_SIDE_IDS]
		joint_vel = self.GetJointVelocities()[RIGHT_SIDE_IDS]
		return np.concatenate((joint_pos, joint_vel))

	def GetFullState2D(self):
		""" Get full robot state:
		-base pos (xz)
		-base orn (p)
		-base linear velocity (dx, dz)
		-base angular velocity (wy)
		-joint pos (4)
		-joint vels (4)
		"""
		return np.concatenate(( self.GetBaseState2D(), self.GetJointState2D() ))

	def DefaultStandUpObs(self,DEBUG_OBS=False):
		""" Default observation for stand up task. Consists of
		- (1 val ) base z
		- (1 val ) base pitch
		- (2 vals) base linvel (dx, dz)
		- (1 vals) base angvel (pitch rate)
		- (4 vals) joint angles
		- (4 vals) joint velocities
		- (4 vals) foot normal forces (in case contacts uneven)
		Total: 17 values

		Note: can also revert observation back to that of the full system and possibly transfer policies across.

		Future observations: may want to include relative feet positions, other forces?
		"""
		obs = []

		# get base pos, orientation, linear and angular velocities
		basePos,baseOrn = self.GetBasePosition(), self.GetBaseOrientationRollPitchYaw()
		linVel, angVel = self.GetBaseLinearVelocity(), self.GetBaseAngularVelocity()
		joint_angles = self.GetJointAngles()
		joint_vels = self.GetJointVelocities()
		_,_, feetNormalForces = self.GetContactInfo()

		obs.extend([basePos[0],basePos[2]])  # only care about z
		obs.extend([baseOrn[1]])
		obs.extend([linVel[0], linVel[2]])
		obs.extend([angVel[1]])
		obs.extend([joint_angles[i] for i in RIGHT_SIDE_IDS])
		obs.extend([joint_vels[i] for i in RIGHT_SIDE_IDS])
		#obs.extend(feetNormalForces)

		if DEBUG_OBS:
			print('z', basePos[2] )
			print('baseOrn', baseOrn)
			print('base linVel', linVel)
			print('base angVel', angVel)
			print('joint angles',joint_angles)
			print('joint vels', joint_vels)
			print('normal forces', feetNormalForces)
		
		return obs
	
	def _defaultObsLimits(self):
		""" get max/min observation limits, somewhat conservative """

		z_min = [-10,0]
		orn_low = [-1]
		linvel_low = [-5]*2
		angvel_low = [-5]
		joint_angles_low = list(LOWER_ANGLE_JOINT[RIGHT_SIDE_IDS])
		joint_vels_low = list(-VELOCITY_LIMITS[RIGHT_SIDE_IDS])
		feetNormalForces_low = [0]*4
		obs_low = z_min + orn_low + linvel_low + angvel_low + joint_angles_low + joint_vels_low #+ feetNormalForces_low

		z_max = [10,5]
		orn_upp = [1]
		linvel_upp = [5]*2
		angvel_upp = [5]
		joint_angles_upp = list(UPPER_ANGLE_JOINT[RIGHT_SIDE_IDS])
		joint_vels_upp = list(VELOCITY_LIMITS[RIGHT_SIDE_IDS])
		feetNormalForces_high = [NORMAL_FORCE_HIGH]*4
		obs_upp = z_max + orn_upp + linvel_upp + angvel_upp + joint_angles_upp + joint_vels_upp #+ feetNormalForces_high

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
		# keep hips at 0
		self.setPD_joints([0]*4,self.hipIds)

	def setInputIK(self,ee_pos_list):
		""" Called with desired end effector list in order of leg ids, which should be output of _scaleActionsToIK """
		raise ValueError("IK for 2D env is broken")
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
		If len(actions) == 2,  interpret as 2 normal forces
		If len(actions) == 4,  interpret as [Fx_front,Fz_front, Fx_rear,Fz_rear]
		If len(actions) == 12, interpret as four XYZ e.e. desired forces (where front pairs and rear pairs should be same), as called from Gym interface

		Force control (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
		"""
		u = ee_forces.copy()
		if len(u) == 2:
			new_u = [0,0,u[0], 0,0,u[0],
					 0,0,u[1], 0,0,u[1]]
			u = np.asarray(new_u)
		elif len(u) == 4:
			new_u = [u[0],0,u[1], u[0],0,u[1],
					 u[2],0,u[3], u[2],0,u[3]]
			u = np.asarray(new_u)
		# compute joint torques (pybullet, vs. i.e. USC software) 
		all_ff_torques_pybullet = np.zeros(12)
		for i in (range(4)):
			J, _ = self._ComputeJacobianAndPositionUSC(i)
			
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
		
	##########################################################################
	## Scale actions from [-1, 1] to ranges for torque control, PD, IK, etc. 
	##########################################################################
	def _scaleActionsToTorques(self,actions):
		""" Scale actions in [-1,1] range to torque limits, clip based on torque limits 
		Assuming getting desired torques for [FR_thigh, FR_calf, RR_thigh, RR_calf]
		"""
		u = TORQUE_LIMITS_2D * np.clip(actions,-1,1)
		full_u = [0, u[0], u[1], 0, u[0], u[1],
				  0, u[2], u[3], 0, u[2], u[3]]
		return np.asarray(full_u)

	def _scaleActionsToPositions(self,actions):
		""" Scale actions in [-1,1] range to position limits, clip based on joint limits 

		Assuming getting [FR_thigh, FR_calf, RR_thigh, RR_calf]
		"""
		# Scaled u based on the following:
		# b is new upper limit,   a is new lower limit
		# max is old upper limit, max is old lower limit
		#        (b-a)(x - min)
		# f(x) = --------------  + a
		#           max - min
		# u = ( ( UPPER_ANGLE_JOINT - LOWER_ANGLE_JOINT ) * (u - (-1) ) ) / (1 - (-1)) + LOWER_ANGLE_JOINT
		# u = LOWER_ANGLE_JOINT +  0.5 * ( UPPER_ANGLE_JOINT - LOWER_ANGLE_JOINT ) * (u + 1) 

		u = np.clip(actions,-1,1)
		u = LOWER_ANGLE_JOINT_2D +  0.5 * ( UPPER_ANGLE_JOINT_2D - LOWER_ANGLE_JOINT_2D ) * (u + 1) 
		u = np.clip(u,LOWER_ANGLE_JOINT_2D,UPPER_ANGLE_JOINT_2D)

		full_u = [0, u[0], u[1], 0, u[0], u[1],
				  0, u[2], u[3], 0, u[2], u[3]]
		return np.asarray(full_u)

	def _scaleActionsToIK(self,actions):
		""" Scale actions in [-1,1] range to corresponding desired x,y,z positions """
		raise ValueError("IK for 2D env is broken")
		u = np.clip(actions,-1,1)
		targetPositions = IK_LOW +  0.5 * ( IK_UPP - IK_LOW ) * (u + 1) 
		return targetPositions

	def _scaleActionsToForces(self,actions):
		""" Scale actions in [-1,1] range to corresponding desired normal forces, to torque commands

		If len(actions) == 2,  interpret as desired forces for front pair, back pair
		If len(actions) == 4, interpret as XZ e.e. desired forces

		Force control (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
		"""
		u = np.clip(actions,-1,1)
		if len(u) == 2:
			# Fz only
			u = MIN_NORMAL_FORCE + 0.5 * (MAX_NORMAL_FORCE - MIN_NORMAL_FORCE) * (u+1)
			u = np.clip(u,MIN_NORMAL_FORCE,MAX_NORMAL_FORCE)
			u_norms_desired = u.copy()
			# copy 0 for desired Fx and Fy 
			new_u = [0,0,u[0], 0,0,u[0],
					 0,0,u[1], 0,0,u[1]]
			u = np.asarray(new_u)
		else:
			# Fx, Fy, Fz at each foot
			u = MIN_XZ_FORCE + 0.5 * (MAX_XZ_FORCE - MIN_XZ_FORCE) * (u+1)
			u = np.clip(u,MIN_XYZ_FORCE,MAX_XYZ_FORCE)
			u_norms_desired = u.copy()[1::2]
			new_u = [u[0],0,u[1], u[0],0,u[1],
					 u[2],0,u[3], u[2],0,u[3]]
			u = np.asarray(new_u)

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
		raise ValueError("IK for 2D Env is broken")
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
												currentPositions=currJointPos,
												residualThreshold=.0001,
												maxNumIterations=200)

		return jointPosesCurr[3*leg_id:3*leg_id+3]

	##########################################################################
	## Internal Step function, which calls Apply/clip action based on mode
	##########################################################################

	##############################################################################
	## Setup quadruped, reset, add noise or perturb states, etc.
	##############################################################################
	def _setUpQuadruped(self):
		"""Reset entire environment, load quadruped. Should be called for any hard reset.
		"""
		self._p.resetSimulation()
		self._p.setTimeStep(self._timeStep)
		self._p.setGravity(0,0,-9.8)
		plane = self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))

		if self._self_collision_enabled:
			self.quadruped = self._p.loadURDF(os.path.join(self._urdfRoot,URDF_FILENAME), 
											[0,0,0.0],[0,0,0,1],flags=self._p.URDF_USE_SELF_COLLISION, useFixedBase=self._useFixedBase)
		else:
			self.quadruped = self._p.loadURDF(os.path.join(self._urdfRoot,URDF_FILENAME), 
											[0,0,0.0],[0,0,0,1], useFixedBase=self._useFixedBase)

		# These are the joints of interest: Hip, Thigh and Calf joints for each of the legs, so total number of ID's are 12
		num_joints = self._p.getNumJoints(self.quadruped)

		self.jointIds=[]
		self.feetIndices = []
		self.hipIds = []
		self.thighIds = []
		self.calfIds = []
		for j in range (num_joints):
			# Remove default joint damping
			self._p.changeDynamics(self.quadruped,j,linearDamping=0, angularDamping=0)
			info = self._p.getJointInfo(self.quadruped,j)
			print(info)
			joint_name = info[1].decode("UTF-8")
			if _BASE_STRING.match(joint_name):
				self.real_base = info[0]
			if _HIP_NAME_PATTERN.match(joint_name):
				self.hipIds.append(j)
				self.jointIds.append(j)
			elif _THIGH_NAME_PATTERN.match(joint_name):
				self.thighIds.append(j)
				self.jointIds.append(j)
			elif _CALF_NAME_PATTERN.match(joint_name):
				self.calfIds.append(j)
				self.jointIds.append(j)
			elif _FOOT_NAME_PATTERN.match(joint_name):
				self.feetIndices.append(j)
			jointType = info[2]
			# prismatic joints are  for the x and z directions
			if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
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

	def ResetBasePositionAndOrientation(self, base_pos, base_orn ):
		""" Resets base to desired x,z,phi position. """
		if len(base_orn) == 4: #quaternion
			base_orn = self._p.getEulerFromQuaternion(base_orn)
		self._p.resetJointState(self.quadruped,0,base_pos[0], targetVelocity=0) # x pos
		self._p.resetJointState(self.quadruped,1,base_pos[2], targetVelocity=0) # z pos
		self._p.resetJointState(self.quadruped,2,base_orn[1], targetVelocity=0) # body pitch
		self._last_set_base_pos = base_pos
		self._last_set_base_orn = base_orn

	def ResetBaseVelocity(self, base_linvel, base_angvel ):
		""" Resets base velocities in x,z,phi rates. 
		MUST be called AFTER ResetBasePositionAndOrientation(). """
		self._p.resetJointState(self.quadruped,0,self._last_set_base_pos[0], targetVelocity=base_linvel[0]) # x pos
		self._p.resetJointState(self.quadruped,1,self._last_set_base_pos[2], targetVelocity=base_linvel[2]) # z pos
		self._p.resetJointState(self.quadruped,2,self._last_set_base_orn[1], targetVelocity=base_angvel[1]) # body pitch

	def SetBaseXZPhi(self,xzphi):
		""" Set base x, z, phi (pitch). """
		basePos = [xzphi[0], 0,xzphi[1]]
		baseOrn = [0, xzphi[2], 0]
		self.ResetBasePositionAndOrientation(basePos,baseOrn)

	# def SetJointStates(self,joint_states):
	# 	"""Take in joint positions and velocities, and set joint states. """
	# 	joint_pos = self.MapRightSideToAllJoints(joint_states[0:4])
	# 	joint_vel = self.MapRightSideToAllJoints(joint_states[4:8])
	# 	for i,jointId in enumerate(self.jointIds):
	# 		self._p.resetJointState(self.quadruped, jointId, joint_pos[i], targetVelocity=joint_vel[i])
	# 	return joint_pos, joint_vel

	def MapRightSideToAllJoints(self,FR_RR):
		""" Take in [FR_thigh, FR_calf, RR_thigh, RR_calf] positions/velocities, 
		and map to corresponding left side positions/velocities. 
		Returns: full 12 joint state (either pos or vel, depending on input).
		"""
		FR = FR_RR[0:2]
		RR = FR_RR[2:4]
		return np.array([ 0, FR[0], FR[1], 0, FR[0], FR[1],
						  0, RR[0], RR[1], 0, RR[0], RR[1]])

	# def SetPoseAndSettle(self,xzphi,joint_states, settle_threshold=0.5):
	# 	""" Set robot base pos, joint states, and simulate until velocities are under some threshold. """
	# 	self.SetBaseXZPhi(xzphi)
	# 	joint_pos, joint_vel = self.SetJointStates(joint_states)
	# 	self.setInputPD(joint_pos)
	# 	self._settle(settle_threshold)

	# 	# make sure the xzphi position is close to what we wanted!
	# 	actual_xzphi = self.GetBaseState2D()[0:3]
	# 	assert np.linalg.norm(xzphi - actual_xzphi) < 0.1


		
	def Reset(self,state="Sit",hardReset=False,addNoise=False):
		""" Reset AlienGo to initial states.
		Can specify state in {"Stand", "Sit", "Zeros"}
		"""
		# load whole quadruped again, avoid since this is slow/unnecessary (usually)
		if self._hard_reset or hardReset:
			self._setUpQuadruped()

		# Note: BASE HAS CHANGED! so the following call is not changing what we want..
		#self._p.resetBasePositionAndOrientation(self.quadruped,[0,0,0.4],[0,0,0,1])
		# ... instead need this
		#self._p.resetJointState(self.quadruped,0,0, targetVelocity=0) # x pos
		#self._p.resetJointState(self.quadruped,1,0.5, targetVelocity=0) # z pos
		#self._p.resetJointState(self.quadruped,2,0, targetVelocity=0) # body pitch
		# leaving above for clarity
		self.SetBaseXZPhi([0,0.5,0])


		# problem with reset - it will keep the last applied motor torques, disable
		self._disable_motor()
		self._step_counter = 0
		# default initial position
		init_position = [0.0, 1.2217304764, -2.77507351067,
						 0.0, 1.2217304764, -2.77507351067,
						 0.0, 1.2217304764, -2.77507351067,
						 0.0, 1.2217304764, -2.77507351067]

		# All zero joint angles, just dangle
		if state == "Zeros":
			start_baseHeight = 0.5
			init_position = [0]*12
			for i, joint_pos in enumerate(init_position):
				p.resetJointState(self.quadruped, self.jointIds[i], joint_pos, targetVelocity=0)

		# Sit (legs drawn in, torso close to ground)
		if state == "Sit":
			start_baseHeight = 0.3
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
			for i, stand_pos in enumerate(stand_init_position):
				self._p.resetJointState(self.quadruped, self.jointIds[i], stand_pos, targetVelocity=0)
			init_position = stand_init_position

		#self._p.resetBasePositionAndOrientation(self.quadruped,[0,0,start_baseHeight],[0,0,0,1])
		self.SetBaseXZPhi([0,start_baseHeight,0])

		# vary number of initial simulations, simulate until get contact
		beg_sims = round(50*np.random.random())
		for _ in range(beg_sims):
			#self.setInputPD(init_position)
			#self.setTorques([0]*12)
			if self._p.getContactPoints() != ():
				# break if already have a contact
				break
			self._p.stepSimulation()
			if self.render:
				time.sleep(1./500.)
		# disable motors again, or else can't do torque control
		self._disable_motor()



		if self.render:
			#self._p.getDebugVisualizerCamera()
			#self.configureVisualizer()
			if self._recordVideo:
				self.RecordVideoHelper()
		
		return


	def RandomInitNoise(self):
		""" initialize robot into random configuration. TODO: make sure no limb is IN the ground. 
		Check valid limits.
		"""
		# These are the current rand limits
		x_limit = y_limit = 0.5
		z_upp = 0.45
		z_low = 0.2
		pitch_limit = 0.5
		# velocity limits
		linvel_limit = 0.5
		pitch_rate_limit = 0.5

		# Set base x,y,z
		x = 2 * (0.5 - np.random.random())
		z = max(z_upp*np.random.random(),z_low)
		# Set roll, pitch, yaw
		pitch = pitch_limit * 2 * (0.5 - np.random.random())
		# set linear velocities
		vx, vz = linvel_limit * 2 * (0.5 - np.random.random(2))
		# set angular velocities
		wy = pitch_rate_limit * 2 * (0.5 - np.random.random())

		baseOrnQuaternion = self._p.getQuaternionFromEuler([0,pitch,0])
		self.ResetBasePositionAndOrientation(self.quadruped, [x,0,z], baseOrnQuaternion)
		self.ResetBaseVelocity(self.quadruped,[vx, 0, vz], [0, wy, 0])

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

		# make 2D
		jointPos = [ 0, jointPos[1], jointPos[2], 0, jointPos[1], jointPos[2],
					 0, jointPos[7], jointPos[8], 0, jointPos[7], jointPos[8]]
		jointVels = [ 0, jointVels[1], jointVels[2], 0, jointVels[1], jointVels[2],
					  0, jointVels[7], jointVels[8], 0, jointVels[7], jointVels[8]]

		# reset joint states
		for i,jointId in enumerate(self.jointIds):
			self._p.resetJointState(self.quadruped, jointId, jointPos[i], targetVelocity=jointVels[i])

	##########################################################################
	## Helper functions for I/O, videos, visualizer, etc.
	##########################################################################

if __name__ == "__main__":
	# test out some functionalities
	control_mode = "PD"
	robot=AlienGo2D(pybullet_client=p,control_mode=control_mode,render=True ,useFixedBase=False)
	print(robot.total_mass)
	#sys.exit()
	while True:
		if control_mode == "TORQUE":
			u = [0,10,0]*4
		elif control_mode=="FORCE":
			u = [robot.total_mass * (-9.8/4)] * 2
			#u = [-56]*4
		elif control_mode=="PD":
			u = [0,0.65, -1.2]*4#, 0.65, -1.2,]
			u = [ 0.000,  1.257, -2.356,  0.000,  1.257, -2.356,  0.000,  1.257, -2.356,  0.000,  1.257, -2.356 ]
		else:
			u = [0]*4
		robot.Step(u)
		_,_,normals = robot.GetContactInfo()
		#print('actual normals', sum(normals), 'expected', robot.total_mass*9.8)
		if robot._terminate():
			robot.Reset()
			#robot.SetPose(None)
		#print(robot.GetBasePosition())
		nicePrint(u)
		nicePrint(robot.GetJointAngles())

		#linJac, angJac = robot._CalculateJacobian(1)
		#J,pos = robot._ComputeJacobianAndPositionUSC(1)
		#nicePrint2D(linJac)
		#nicePrint2D(J)
		#robot.GetLocalEEPos(0)