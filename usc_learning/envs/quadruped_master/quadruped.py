"""This file implements the functionalities of a quadruped using pybullet.
Should subclass for specific elements for aliengo, laikago, minitaur, rex.
DEFAULT is: Laikago (due to debugged copy, and closer to aliengo/a1 than minitaur)

"""
import os
import copy
import math
import numpy as np
import time
import re
import usc_learning.envs.pyb_data as pyb_data
from usc_learning.envs.quadruped_master import quadruped_motor
from usc_learning.utils.robot_utils import FormatFRtoAll

""" minitaur """
# overheat protection?
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
BASE_LINK_ID = -1

INCLUDE_TORQUES_IN_ACTION_SPACE = False
BODY_XY_IN_OBS = True # NOTE: just Y for now 
MAX_BODY_LIN_VEL = 20
MAX_BODY_ANG_VEL = 20

# general, may not be updated, check quadruped_gym_env.py
CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD", "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE"]# , "IMPEDANCE_FORCE"]

class Quadruped(object):
  """The quadruped class to simulate Aliengo, A1, or Laikago.
  """
  def __init__(self,
               pybullet_client,
               robot_config=None,
               time_step=0.01,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=True,
               motor_control_mode="PD",
               motor_overheat_protection=False,
               #rl_action_high=1.0,
               on_rack=False,
               render=False):
               #kd_for_pd_controllers=0.3):
    """Constructs a quadruped and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      robot_config: File with all of the relevant configs for the desired 
        quadruped.
      time_step: The time step of the simulation.
      self_collision_enabled: Whether to enable self collision.
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_control_mode: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in quadruped.py for more
        details.
      rl_action_high: ouput high from RL, used for scaling
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      kd_for_pd_controllers: kd value for the pd controllers of the motors.
    """
    self._robot_config = robot_config
    self.num_motors = self._robot_config.NUM_MOTORS
    self.num_legs = self._robot_config.NUM_LEGS 
    self._pybullet_client = pybullet_client
    self._urdf_root = self._robot_config.URDF_ROOT # urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._pd_control_enabled = pd_control_enabled
    self._motor_direction = self._robot_config.JOINT_DIRECTIONS
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    #self._rl_action_high = rl_action_high
    #self._max_force = 3.5
    self._accurate_motor_model_enabled = accurate_motor_model_enabled

    # motor control mode for accurate motor model, should only be torque or position at this low level
    if motor_control_mode in ["PD","IK", "PMTG_JOINT", "PMTG_XY_ONLY_JOINT", "BEZIER_GAIT_JOINT", "PMTG_XYZ_ONLY_JOINT", "PMTG_FULL_JOINT"]:
      self._motor_control_mode = "PD"
    elif motor_control_mode in ["TORQUE","FORCE","IMPEDANCE","IMPEDANCE_POS_ONLY","IMPEDANCE_POS_GAINS", "IMPEDANCE_POS_ONLY_IK", "IMPEDANCE_POS_ONLY_SPHERICAL", "IMPEDANCE_POS_ONLY_SPHERICAL_IK",
                    "IMPEDANCE_POS_VEL_ONLY", "PMTG", "PMTG_XY_ONLY", "PMTG_XYZ_ONLY", "PMTG_FULL",# "IMPEDANCE_FORCE",
                    "IMPEDANCE_GAINS","IMPEDANCE_3D_FORCE", "IMPEDANCE_FULL", "IMPEDANCE_NO_Y", "IMPEDANCE_XZ_ONLY","IMPEDANCE_TROT", "BEZIER_GAIT"]:
      self._motor_control_mode = "TORQUE"
    else:
      raise ValueError('motor control mode unclear')

    self._sensory_hist_len = 2
    self._sensor_records = None
    # self._sensor_records = [np.zeros(73)] * self._sensory_hist_len  # WE USE IT
    #self._motor_control_mode = motor_control_mode
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self.render = render
    if self._accurate_motor_model_enabled:
      self._kp = self._robot_config.MOTOR_KP #motor_kp
      self._kd = self._robot_config.MOTOR_KD
      self._motor_model = quadruped_motor.QuadrupedMotorModel(
                                    motor_control_mode=self._motor_control_mode,
                                    kp=self._kp,
                                    kd=self._kd,
                                    torque_limits=self._robot_config.TORQUE_LIMITS)
    elif self._pd_control_enabled:
      self._kp = self._robot_config.MOTOR_KP #8
      self._kd = self._robot_config.MOTOR_KD # kd_for_pd_controllers
      self._motor_model = None
    else:
      self._kp = 1
      self._kd = 1
      self._motor_model = None
    self.time_step = time_step
    self.Reset(reload_urdf=True)

    self.save_xyz = []
    self.save_Fn = []
    self.save_real_xyz = []
    self.save_real_Fn = []

  ######################################################################################
  # Robot states and observation related
  ######################################################################################
  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return self._robot_config.INIT_RACK_POSITION
    else:
      return self._robot_config.INIT_POSITION

  def _GetDefaultInitOrientation(self):
    return self._robot_config.INIT_ORIENTATION

  def GetBasePosition(self):
    """Get the position of quadruped's base.

    Returns:
      The position of quadruped's base.
    """
    position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetBaseOrientation(self):
    """Get the orientation of quadruped's base, represented as quaternion.

    Returns:
      The orientation of quadruped's base.
    """
    _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    # for laikago, it is not upright to begin with, for other robots the inv is just identity
    #print('orientation',orientation)
    #print('inv', self._robot_config.INIT_ORIENTATION_INV)
    _, orientation = self._pybullet_client.multiplyTransforms(
      positionA=[0, 0, 0],
      orientationA=orientation,
      positionB=[0, 0, 0],
      orientationB=self._robot_config.INIT_ORIENTATION_INV)
    return orientation

  def GetBaseState(self):
    """ Get base state: xyz, rpy, dxyz, drpy  """
    basePos = self.GetBasePosition()
    baseRPY = self.GetBaseOrientationRollPitchYaw()
    baseLinVel = self.GetBaseLinearVelocity()
    baseAngVel = self.GetBaseAngularVelocity()
    return np.concatenate((basePos, baseRPY, baseLinVel, baseAngVel))

  def GetBaseOrientationRollPitchYaw(self):
    """Get quadruped's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame.
    """
    orientation = self.GetBaseOrientation()
    # this results in angles only in [-pi/2, pi/2], test with atan2 
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)

    """
    # test with atan2 
    #// roll (x-axis rotation)
    x,y,z,w = orientation
    sinr_cosp = 2 * (w * x + y * z);
    cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = np.arctan2(sinr_cosp, cosr_cosp);

    #// pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x);
    if abs(sinp) >= 1:
        print('sinp', sinp)
        pitch = np.sign(sinp) * np.pi/2 # std::copysign(M_PI / 2, sinp) #; // use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    #// yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y);
    cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = np.arctan2(siny_cosp, cosy_cosp);

    roll_pitch_yaw = [roll, pitch, yaw]
    """

    return np.asarray(roll_pitch_yaw)

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
    _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0], orientation)
    # Transform the angular_velocity at neutral orientation using a neutral
    # translation and reverse of the given orientation.
    relative_velocity, _ = self._pybullet_client.multiplyTransforms([0, 0, 0], 
                            orientation_inversed, 
                            angular_velocity,
                             self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
    return np.asarray(relative_velocity)

  def GetBaseOrientationMatrix(self):
    """ Get the base orientation matrix, as numpy array """
    baseOrn = self.GetBaseOrientation()
    return np.asarray(self._pybullet_client.getMatrixFromQuaternion(baseOrn)).reshape((3,3))

  def GetBaseLinearVelocity(self):
    """ Get base linear velocities (dx, dy, dz) """
    linVel,_ = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray(linVel)

  def GetBaseAngularVelocity(self):
    """ Get base angular velocities (droll, dpitch, dyaw) """
    _, angVel = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray(angVel)

  def GetMotorAngles(self):
    """Get quadruped motor angles at the current moment.

    Returns:
      Motor angles.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorVelocities(self):
    """Get the velocity of all motors.

    Returns:
      Velocities of all motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorTorques(self):
    """Get the torques the motors are exerting.

    Returns:
      Motor torques of all motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques


  def GetLocalEEPos(self,legID):
    """ Find end effector position of legID in body frame. """
    link_state = self._pybullet_client.getLinkState(self.quadruped,self._foot_link_ids[legID],
                                              computeLinkVelocity=True, computeForwardKinematics=True)
    linkPos = link_state[4]
    linkOrn = link_state[5]
    # map the world location of the link into local coordinates of the robot body (and then into hip/leg frame)
    basePos, baseOrn = self.GetBasePosition(), self.GetBaseOrientation() 
    invBasePos, invBaseOrn = self._pybullet_client.invertTransform(basePos,baseOrn)
    linkPosInBase, linkOrnInBase = self._pybullet_client.multiplyTransforms(invBasePos, invBaseOrn, linkPos, linkOrn)
    linkPosInBase = np.asarray(linkPosInBase)
    return linkPosInBase

  def GetContactInfo(self):
    """ Returns 
    # (1) num valid contacts (only those between a foot and the ground)
    # (2) num invalid contacts (either between a limb and the ground (other than the feet) or any self collisions
    # (3) normal force at each foot (0 if in the air) 
    # (4) bool if each foot in contact (1) or not (0)
    """
    # penalize this in reward function ( or terminate condition ? )
    numValidContacts = 0
    numInvalidContacts = 0
    feetNormalForces = [0,0,0,0]
    feetInContactBool = [0,0,0,0]
    # print('=========================== contacts')
    for c in self._pybullet_client.getContactPoints():
      # print('c', c)
      # print('thigh ids', self._thigh_ids)
      # print('calf ids', self._calf_ids)
      # print('foot ids', self._foot_link_ids)
      # print(c[3] in self._calf_ids)
      # print(c[4] in self._calf_ids)
      # if bodyUniqueIdA is same as bodyUniqueIdB, self collision
      if c[1] == c[2]:
        # , but only check calves for now
        if ( (c[3] in self._calf_ids ) or (c[4] in self._calf_ids)):
          numInvalidContacts += 1
        continue
      # if thighs in contact with anything, this is a failing condition
      if c[3] in self._thigh_ids or c[4] in self._thigh_ids:
        numInvalidContacts += 1
        continue
      # check link indices, one MUST be a foot index, and the other must be -1 (plane)
      # depending on sim and pybullet version, this can happen in either order (apparently)
      if ( c[3] == -1 and c[4] not in self._foot_link_ids ) or \
         ( c[4] == -1 and c[3] not in self._foot_link_ids ):
        numInvalidContacts += 1
      else:
        numValidContacts += 1
        try:
          footIndex = self._foot_link_ids.index(c[4])
        except:
          footIndex = self._foot_link_ids.index(c[3])
        feetNormalForces[footIndex] += c[9] # if multiple contact locations?
        feetInContactBool[footIndex] = 1

    # # test just to make sure collisions with non-ground contacts are ok
    # for c in self._pybullet_client.getContactPoints():
    #   # if bodyUniqueIdA is same as bodyUniqueIdB, self collision
    #   if c[1] == c[2]:
    #     numInvalidContacts += 1
    #     continue
    #   # check link indices, one MUST be a foot index, and the other must be -1 (plane)
    #   # depending on sim and pybullet version, this can happen in either order (apparently)
    #   if ( c[3] in self._foot_link_ids ):
    #     footIndex = self._foot_link_ids.index(c[3])
    #     feetInContactBool[footIndex] = 1
    #   if ( c[4] in self._foot_link_ids ):
    #     footIndex = self._foot_link_ids.index(c[4])
    #     feetInContactBool[footIndex] = 1
    # print(feetInContactBool)
    # print('contacts',feetInContactBool)
    # print(numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool)
    # print('foot links', self._foot_link_ids)
    return numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool

  def ThighsInContact(self):
    """ Check if thighs in contact (for termination condition)
    """
    # print('=========================== contacts')
    for c in self._pybullet_client.getContactPoints():
      # if thighs in contact with anything, this is a failing condition
      if c[3] in self._thigh_ids or c[4] in self._thigh_ids:
        return True
    return False

  def GetFootContactForces(self):
    """ Returns 
    xyz contact forces for each foot 
    """
    # penalize this in reward function ( or terminate condition ? )
    footForces = np.zeros(12)
    numValidContacts = 0
    numInvalidContacts = 0
    feetNormalForces = [0,0,0,0]
    feetInContactBool = [0,0,0,0]

    #print('=========================== contacts')
    for c in self._pybullet_client.getContactPoints():
      #print('c', c)
      # normal force * contact direction 
      #print('contact force', c[9]*np.array(c[7]))
      # if bodyUniqueIdA is same as bodyUniqueIdB, self collision
      if c[1] == c[2]:
        numInvalidContacts += 1
        continue
      # check link indices, one MUST be a foot index, and the other must be -1 (plane)
      # depending on sim and pybullet version, this can happen in either order (apparently)
      if ( c[3] == -1 and c[4] not in self._foot_link_ids ) or \
         ( c[4] == -1 and c[3] not in self._foot_link_ids ):
        numInvalidContacts += 1
      else:
        numValidContacts += 1
        try:
          footIndex = self._foot_link_ids.index(c[4])
        except:
          footIndex = self._foot_link_ids.index(c[3])
        feetNormalForces[footIndex] += c[9] # if multiple contact locations?
        feetInContactBool[footIndex] = 1
        footForces[footIndex*3:footIndex*3+3] += c[9]*np.array(c[7]) # if multiple contacts?  would be technically wrong 

    return footForces

  ######################################################################################
  # Observation - joint info and orientation only [ORIGINAL]
  ######################################################################################
  def GetObservation(self):
    """Get the observations of quadruped.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. 
      observation[0:12] are motor angles. 
      observation[12:24] are motor velocities, 
      observation[24:36] are motor torques.
      observation[36:40] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.GetMotorAngles().tolist())
    observation.extend(self.GetMotorVelocities().tolist())
    if INCLUDE_TORQUES_IN_ACTION_SPACE:
      observation.extend(self.GetMotorTorques().tolist())
    observation.extend(list(self.GetBaseOrientation()))
    return observation

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.array([0.0] * len(self.GetObservation()))
    #upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    #upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    #upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
    upper_bound[0:self.num_motors] = (self._robot_config.UPPER_ANGLE_JOINT )#np.pi  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    if INCLUDE_TORQUES_IN_ACTION_SPACE:
      upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
      upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    else:
      upper_bound[-4:] = 1.0 # Quaternion of base orientation.
    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    #return -self.GetObservationUpperBound()
    lower_bound = np.array([0.0] * len(self.GetObservation()))
    #upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    #upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    #upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
    lower_bound[0:self.num_motors] = (self._robot_config.LOWER_ANGLE_JOINT )#np.pi  # Joint angle.
    lower_bound[self.num_motors:2 * self.num_motors] = (-self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    if INCLUDE_TORQUES_IN_ACTION_SPACE:
      lower_bound[2 * self.num_motors:3 * self.num_motors] = (-self._robot_config.TORQUE_LIMITS)  # Joint torque.
      lower_bound[3 * self.num_motors:] = -1.0  # Quaternion of base orientation.
    else:
      lower_bound[-4:] = -1.0 # Quaternion of base orientation 
    return lower_bound

  ######################################################################################
  # Observation - include contacts, z height, base linear velocity 
  ######################################################################################
  def GetObservationContacts(self):
    """Get the observations of quadruped. Include contact data (bool) for feet. Also add z.
    Returns:
      GetObservation() [motor (angles, velocities, torques), base orientation] 
      Base z height
      Contact data booleans for feet in contact
      Base linear velocities (in robot frame? but training to maximize vel in global x direction)
    """
    observation = self.GetObservation()
    # basePos = 
    # print('\n',basePos)
    if BODY_XY_IN_OBS:
      observation.extend(self.GetBasePosition()[1:])
    else:
      observation.extend([self.GetBasePosition()[2]])
    _,_,_,feetInContactBool = self.GetContactInfo()
    #R = self.GetBaseOrientationMatrix()
    observation.extend(list(self.GetBaseLinearVelocity()))
    observation.extend(list(self.GetTrueBaseRollPitchYawRate()))
    #print(feetInContactBool)

    # add foot pos/ foot vel 
    for i in range(4):
      #foot_pos = self.GetLocalEEPos(i)
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
      # foot velocity in leg frame
      foot_linvel = J @ self.GetMotorVelocities()[i*3:i*3+3]
      # observation.extend(tuple(foot_pos))
      observation.extend(tuple(ee_pos_legFrame))
      observation.extend(tuple(foot_linvel))


    # add des foot position 
    #observation.extend(self._des_foot_pos)

    # add foot forces
    # observation.extend(self.GetFootContactForces())

    # add last action from RL 
    # observation.extend(self._last_action_rl)

    observation.extend(feetInContactBool)

    return observation

  def GetObservationContactsUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    # upper_bound = np.array([0.0] * len(self.GetObservationContacts()))
    # upper_bound[0:3*self.num_motors+4] = self.GetObservationUpperBound()
    # upper_bound[3*self.num_motors+4] = 1.0  # Body z .
    # upper_bound[3*self.num_motors+5:3*self.num_motors+5+4] = [1.]*4  # In contact bools
    # # upper_bound[3*self.num_motors+9:] = (4.,4.,4.) # base lin vel max
    # upper_bound[3*self.num_motors+9:3*self.num_motors+9+6] = (10.,10.,10.,10.,10.,10.) # base lin/ang vel max
    # # foot pos/vel
    # upper_bound[-24:] = np.array([1,1,1,10,10,10]*4 )

    upper_bound = np.concatenate(( self.GetObservationUpperBound(), 
                                  np.array([1.0]) , # body z
                                  # np.array([10.,10.,10.,10.,10.,10.]), # base lin/ang vel max
                                  np.array([MAX_BODY_LIN_VEL]*3), # base lin vel max
                                  np.array([MAX_BODY_ANG_VEL]*3), # base ang vel max
                                  np.array([1,1,1,10,10,10]*4 ), # foot pos/vel
                                  #np.array([1,1,1]*4 ), # des foot pos
                                  # np.array([100]*12 ), # foot forces
                                  np.array([1,1,1]*4 ), # last action rl
                                  np.array([1.]*4), # in contact bools
                                  ))

    if BODY_XY_IN_OBS:
      upper_bound = np.concatenate(( self.GetObservationUpperBound(), 
                                  np.array([100,1.0]) , # body z
                                  np.array([MAX_BODY_LIN_VEL]*3), # base lin vel max
                                  np.array([MAX_BODY_ANG_VEL]*3), # base ang vel max
                                  np.array([1,1,1,10,10,10]*4 ), # foot pos/vel
                                  #np.array([1,1,1]*4 ), # des foot pos
                                  # np.array([100]*12 ), # foot forces
                                  np.array([1.]*4), # in contact bools
                                  ))

    return upper_bound

  def GetObservationContactsLowerBound(self):
    """Get the lower bound of the observation.

    Returns:
      The lower bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    # lower_bound = np.array([0.0] * len(self.GetObservationContacts()))
    # lower_bound[0:3*self.num_motors+4] = self.GetObservationLowerBound()
    # lower_bound[3*self.num_motors+4] = 0. #self._robot_config.IS_FALLEN_HEIGHT  # Body z .
    # lower_bound[3*self.num_motors+5:3*self.num_motors+5+4] = [0.]*4  # In contact bools
    # # lower_bound[3*self.num_motors+9:] = (-4.,-4.,-4.) # base lin vel max
    # lower_bound[3*self.num_motors+9:3*self.num_motors+9+6] = (-10.,-10.,-10.,-10.,-10.,-10.) # base lin/ang vel min

    # # foot pos/vel
    # lower_bound[-24:] = np.array([-1,-1,-1,-10,-10,-10]*4 )

    lower_bound = np.concatenate(( self.GetObservationLowerBound(), 
                                  np.array([0.]) , # body xyz
                                  # np.array([-10.,-10.,-10.,-10.,-10.,-10.]), # base lin/ang vel max
                                  np.array([-MAX_BODY_LIN_VEL]*3), # base lin vel max
                                  np.array([-MAX_BODY_ANG_VEL]*3), # base ang vel max
                                  np.array([-1,-1,-1,-10,-10,-10]*4 ), # foot pos/vel
                                  #np.array([-1,-1,-1]*4 ), # des foot pos
                                  # np.array([-100]*12 ), # foot forces
                                  np.array([-1,-1,-1]*4 ), # last action rl
                                  np.array([0.]*4), # in contact bools
                                  ))

    if BODY_XY_IN_OBS:
      lower_bound = np.concatenate(( self.GetObservationLowerBound(), 
                                  np.array([-100, 0.]) , # body xyz
                                  np.array([-MAX_BODY_LIN_VEL]*3), # base lin vel max
                                  np.array([-MAX_BODY_ANG_VEL]*3), # base ang vel max
                                  np.array([-1,-1,-1,-10,-10,-10]*4 ), # foot pos/vel
                                  #np.array([-1,-1,-1]*4 ), # des foot pos
                                  # np.array([-100]*12 ), # foot forces
                                  np.array([0.]*4), # in contact bools
                                  ))

    return lower_bound

  ######################################################################################
  # Extended Observation
  ######################################################################################
  def GetExtendedObservation(self):
    """Get the full observation of quadruped, includes body positions and velocities.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. 
      observation[0:12] are motor angles. 
      observation[12:24] are motor velocities, 
      observation[24:36] are motor torques.
      observation[36:40] is the orientation of the base, in quaternion form.
      observation[40:43] is the base position
      observation[43:46] is base linear velocity
      observation[46:49] is base angular velocity
    """
    observation = []
    observation.extend(self.GetMotorAngles().tolist())
    observation.extend(self.GetMotorVelocities().tolist())
    observation.extend(self.GetMotorTorques().tolist())
    observation.extend(list(self.GetBaseOrientation()))
    observation.extend(list(self.GetBasePosition()))
    observation.extend(list(self.GetBaseLinearVelocity()))
    observation.extend(list(self.GetTrueBaseRollPitchYawRate()))
    return observation



  def GetExtendedObservationUpperBound(self):
    """Get the upper bound of the observation.
    Note: the upper/lower bounds are going to be larger than actual

    Returns:
      The upper bound of an observation. See GetExtendedObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.array([0.0] * len(self.GetExtendedObservation()))
    upper_bound[0:self.num_motors] = (self._robot_config.UPPER_ANGLE_JOINT)  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
    upper_bound[3 * self.num_motors:3 * self.num_motors+4] = 1.0  # Quaternion of base orientation.
    upper_bound[3 * self.num_motors+4:3 * self.num_motors+4+3] = (4.,2.,2.) # base pos max
    # upper_bound[3 * self.num_motors+4+3:3 * self.num_motors+4+6] = (4.,4.,4.) # base lin vel max
    # upper_bound[3 * self.num_motors+4+6:3 * self.num_motors+4+9] = (4.,4.,4.) # base ang vel max
    upper_bound[3 * self.num_motors+4+3:3 * self.num_motors+4+6] = (10.,10.,10.) # base lin vel max
    upper_bound[3 * self.num_motors+4+6:3 * self.num_motors+4+9] = (10.,10.,10.) # base ang vel max
    return upper_bound

  def GetExtendedObservationLowerBound(self):
    """Get the lower bound of the observation."""
    lower_bound = np.array([0.0] * len(self.GetExtendedObservation()))
    lower_bound[0:self.num_motors] = (self._robot_config.LOWER_ANGLE_JOINT)  # Joint angle.
    lower_bound[self.num_motors:2 * self.num_motors] = (-self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    lower_bound[2 * self.num_motors:3 * self.num_motors] = (-self._robot_config.TORQUE_LIMITS)  # Joint torque.
    lower_bound[3 * self.num_motors:3 * self.num_motors+4] = -1.0  # Quaternion of base orientation.
    lower_bound[3 * self.num_motors+4:3 * self.num_motors+4+3] = (-2.,-2.,0.1) # base pos min
    # lower_bound[3 * self.num_motors+4+3:3 * self.num_motors+4+3+3] = (-4.,-4.,-4.) # base lin vel min
    # lower_bound[3 * self.num_motors+4+6:3 * self.num_motors+4+9] = (-4.,-4.,-4.) # base ang vel min
    lower_bound[3 * self.num_motors+4+3:3 * self.num_motors+4+3+3] = (-10.,-10.,-10.) # base lin vel min
    lower_bound[3 * self.num_motors+4+6:3 * self.num_motors+4+9] = (-10.,-10.,-10.) # base ang vel min
    return lower_bound

  ######################################################################################
  # Extended Observation 2
  ######################################################################################
  def GetExtendedObservation2(self):
    """Get the full observation of quadruped, includes body positions and velocities.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. 
      observation[0:12] are motor angles. 
      observation[12:24] are motor velocities, 
      observation[24:36] are motor torques.
      observation[36:40] is the orientation of the base, in quaternion form.
      observation[40:43] is the base position
      observation[43:46] is base linear velocity
      observation[46:49] is base angular velocity
    """
    observation = self.GetExtendedObservation()
    # print("observation in:", len(observation)) # 49
    # add foot pos/ foot vel 
    for i in range(4):
      foot_pos = self.GetLocalEEPos(i)
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
      # foot velocity in leg frame
      foot_linvel = J @ self.GetMotorVelocities()[i*3:i*3+3]
      observation.extend(tuple(foot_pos)) # 12
      observation.extend(tuple(foot_linvel)) # 12

    # print("observation size:", len(observation))
    return observation # size now is 49+ 24= 73

  def GetExtendedObservation2_History(self, **kwargs):

    base_observation = self.GetExtendedObservation2() # has size of 73

    if "des_vel" in kwargs.keys():
      base_observation = np.append(base_observation, kwargs["des_vel"])
      if self._sensor_records is None:
        self._sensor_records = [np.zeros(74)] * self._sensory_hist_len
    elif self._sensor_records is None:
      self._sensor_records = [np.zeros(73)] * self._sensory_hist_len
    self._sensor_records.pop(0)
    self._sensor_records.append(base_observation)
    # sensor_observation = np.stack(self._sensor_records, axis=0).flatten()
    sensor_observation = np.concatenate(self._sensor_records)
    # print("sensor observation size:", sensor_observation.shape)
    return sensor_observation

    # return base_observation


  def GetExtendedObservation2UpperBound(self):
    """Get the upper bound of the observation.
    Note: the upper/lower bounds are going to be larger than actual

    Returns:
      The upper bound of an observation. See GetExtendedObservation() for the details
        of each element of an observation.
    """
    upper_bound = self.GetExtendedObservationUpperBound() # size is 49
    # print("upper_bound size:", upper_bound.shape) # 49
    # fake bounds, need to calculate
    upper_bound = np.concatenate(( upper_bound, 
                                  np.array([1,1,1,10,10,10]*4 ))) # size is 49+24=73

    # print("upper_bound size:", upper_bound.shape) # 73
    return upper_bound

  def GetExtendedObservation2LowerBound(self):
    """Get the lower bound of the observation."""
    lower_bound = self.GetExtendedObservationLowerBound()
    lower_bound = np.concatenate(( lower_bound, 
                                  np.array([-1,-1,-1,-10,-10,-10]*4 )))
    # print("upper_bound size", lower_bound.shape) # 73
    return lower_bound

  def GetExtendedObservation2UpperBound_History(self):
    """Get the upper bound of the observation.
    Note: the upper/lower bounds are going to be larger than actual

    Returns:
      The upper bound of an observation. See GetExtendedObservation() for the details
        of each element of an observation.
    """
    upper_bound = self.GetExtendedObservationUpperBound() # size is 49
    # print("upper_bound size:", upper_bound.shape) # 49
    # fake bounds, need to calculate
    upper_bound = np.concatenate(( upper_bound,
                                  np.array([1,1,1,10,10,10]*4 ))) # size is 49+24=73
    upper_bound = [upper_bound] * self._sensory_hist_len

    upper_bound = np.concatenate(upper_bound)

    # print("upper_bound size:", upper_bound.shape) # 73*hist_len

    return upper_bound

  def GetExtendedObservation2LowerBound_History(self):
    """Get the lower bound of the observation."""
    lower_bound = self.GetExtendedObservationLowerBound()
    lower_bound = np.concatenate(( lower_bound,
                                  np.array([-1,-1,-1,-10,-10,-10]*4 )))

    lower_bound = [lower_bound] * self._sensory_hist_len

    lower_bound = np.concatenate(lower_bound)

    # print("lower_bound size:", lower_bound.shape) # 73*hist_len
    return lower_bound


  ######################################################################################
  # Extended Observation 2 W CAMERA
  ######################################################################################
  def GetExtendedObservation2Vision(self):
    """Get the full observation of quadruped, as well as vision information (sampled). 

    """
    observation = self.GetObservationContacts() 

    # add foot pos/ foot vel 
    for i in range(4):
      foot_pos = self.GetLocalEEPos(i)
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
      # foot velocity in leg frame
      foot_linvel = J @ self.GetMotorVelocities()[i*3:i*3+3]
      observation.extend(tuple(foot_pos))
      observation.extend(tuple(foot_linvel))

    _, depthImg = self.get_cam_view() # these are 300 by 400
    #rgbImg = np.array(rgbImg)
    depthImg = np.array(depthImg)
    # print(rgbImg.shape,depthImg.shape)
    sampledDepth = depthImg[::20,::20] # so now 15 by 20
    observation.extend(list(sampledDepth.reshape(-1)))

    return observation

  def GetExtendedObservation2VisionUpperBound(self):
    """Get the upper bound of the observation.
    Note: the upper/lower bounds are going to be larger than actual

    Returns:
      The upper bound of an observation. See GetExtendedObservation() for the details
        of each element of an observation.
    """
    upper_bound = self.GetObservationContactsUpperBound()
    # fake bounds, need to calculate
    # foot pos/vel
    upper_bound = np.concatenate(( upper_bound, 
                                  np.array([1,1,1,10,10,10]*4 )))
    # camera depth info
    upper_bound = np.concatenate(( upper_bound, 
                                  np.ones(300) ))
    return upper_bound

  def GetExtendedObservation2VisionLowerBound(self):
    """Get the lower bound of the observation."""
    lower_bound = self.GetObservationContactsLowerBound()
    # foot pos/vel
    lower_bound = np.concatenate(( lower_bound, 
                                  np.array([-1,-1,-1,-10,-10,-10]*4 )))
    # camera depth info
    lower_bound = np.concatenate(( lower_bound, 
                                  np.zeros(300)))
    return lower_bound

  # def GetObservationDimension(self):
  #   """Get the length of the observation list.

  #   Returns:
  #     The length of the observation list.
  #   """
  #   return len(self.GetObservation())

  ######################################################################################
  # INPUTS: set torques, ApplyAction, etc.
  ######################################################################################
  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle): 
    # get max_force 
    # idx = np.where(self._joint_ids == motor_id)
    # print(idx,self._joint_ids, motor_id)
    max_force = self._robot_config.TORQUE_LIMITS[self._joint_ids.index(motor_id)]
    max_vel = self._robot_config.VELOCITY_LIMITS[self._joint_ids.index(motor_id)]
    # print(max_force)
    # max_force = 1000
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.POSITION_CONTROL,
                                                targetPosition=desired_angle,
                                                # positionGain=self._kp,
                                                # velocityGain=self._kd,
                                                # force=self._max_force,
                                                force=max_force,
                                                maxVelocity=max_vel,
                                                )

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name], desired_angle)

  def _ApplyOverheatProtection(self, actual_torque):
    """ Leaving for now, possibly use for aliengo? """
    if self._motor_overheat_protection:
      for i in range(self.num_motors):
        if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
          self._overheat_counter[i] += 1
        else:
          self._overheat_counter[i] = 0
        if (self._overheat_counter[i] >
            OVERHEAT_SHUTDOWN_TIME / self.time_step):
          self._motor_enabled_list[i] = False

  def ApplyAction(self, motor_commands):
    """Set the desired motor angles to the motors of the quadruped.

    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

    Args:
      motor_commands: The desired motor angles.
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetMotorAngles()
      motor_commands_max = (current_motor_angle + self.time_step * self._motor_velocity_limit)
      motor_commands_min = (current_motor_angle - self.time_step * self._motor_velocity_limit)
      motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      q = self.GetMotorAngles()
      qdot = self.GetMotorVelocities()

      
      if 0 and self._motor_control_mode in ["PD", "IK", "PMTG_JOINT","PMTG_XY_ONLY_JOINT"]: # also for IK, anything ultimately outputting positions
        """ clamp motor commands by a max delta, in case weird things happen """
        # here this is clipping twice  - first is arbitrary, second is physics based,
        max_angle_change = self._robot_config.MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        # print('q', q, 'max change', max_angle_change)
        # print('motor commands', motor_commands)
        motor_commands = np.clip(motor_commands,
                               q - max_angle_change,
                               q + max_angle_change)
        # real clip
        motor_commands_min = (q - self.time_step * self._robot_config.VELOCITY_LIMITS)
        motor_commands_max = (q + self.time_step * self._robot_config.VELOCITY_LIMITS)
        motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)

      """ HACK for hips to be 0 """
      #motor_commands[0::3] = 0

      #print('MOTOR COMMANDS \n', motor_commands, '\n')
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(
            motor_commands, q, qdot)
        # May turn off the motor
        #self._ApplyOverheatProtection(actual_torque)

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)

        #print('apply action torque', self._applied_motor_torque )

        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                         self._applied_motor_torque,
                                                         self._motor_enabled_list):
          if motor_enabled:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
      else:
        raise ValueError("Should be using accurate motor model.")
        torque_commands = -self._kp * (q - motor_commands) - self._kd * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list, self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)
    else:
      # raise ValueError("Should be using accurate motor model.")
      motor_commands_with_direction = np.multiply(motor_commands, self._motor_direction)
      for motor_id, motor_command_with_direction in zip(self._motor_id_list,
                                                        motor_commands_with_direction):
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)


  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.

    Args:
      actions: The theta, phi of the leg model.
    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """

    motor_angle = copy.deepcopy(actions)
    scale_for_singularity = 1
    offset_for_singularity = 1.5
    half_num_motors = int(self.num_motors / 2)
    quater_pi = math.pi / 4
    for i in range(self.num_motors):
      action_idx = i // 2
      forward_backward_component = (
          -scale_for_singularity * quater_pi *
          (actions[action_idx + half_num_motors] + offset_for_singularity))
      extension_component = (-1)**i * quater_pi * actions[action_idx]
      if i >= half_num_motors:
        extension_component = -extension_component
      motor_angle[i] = (math.pi + forward_backward_component + extension_component)
    return motor_angle
  ######################################################################################
  # Scale actions, different action spaces, etc.
  ######################################################################################
  def _scale_helper(self, action, lower_lim, upper_lim):
    """Helper to linearly scale from [-1,1] to lower/upper limits. 
    This needs to be made general in case action range is not [-1,1]
    """
    new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
    # verify clip
    new_a = np.clip(new_a, lower_lim, upper_lim)
    return new_a

  def ScaleActionToImpedanceControlSpherical(self,actions,useJointPD=False):
    """Scale action from [-1,1] to spherical coordinates, then to foot xyz position 
    r in [0.15,0.35], theta in [-0.8,0.8], psi in [-0.8,0.8]
    """
    low_lim = np.array([0.15, -0.8, -0.8]*4)
    upp_lim = np.array([0.35,  0.8,  0.8]*4)
    u = np.clip(actions,-1,1)
    u = self._scale_helper(u, low_lim, upp_lim) 

    action = np.zeros(12)

    for i in range(4):
      spherical_coords = self.spherical2cartesian(u[3*i],u[3*i+1],u[3*i+2])
      if i==0 or i==2:
        hip = np.array([0, -self._robot_config.HIP_LINK_LENGTH, 0])
      else:
        hip = np.array([0,  self._robot_config.HIP_LINK_LENGTH, 0])

      xyz = np.array([1,1,-1])*spherical_coords + hip

      tau = self._ComputeLegImpedanceControlActions(xyz,
                                              np.zeros(3),
                                              self._robot_config.kpCartesian,
                                              self._robot_config.kdCartesian,
                                              np.zeros(3),
                                              i)

      action[3*i:3*i+3] = tau

      if useJointPD:
        # test with adding joint PD as well 
        # current angles
        q = self.GetMotorAngles()[3*i:3*i+3]
        dq = self.GetMotorVelocities()[3*i:3*i+3]
        kps = self._robot_config.MOTOR_KP[3*i:3*i+3]
        kds = self._robot_config.MOTOR_KD[3*i:3*i+3]
        # corresponding desired joint angles:
        q_des = self._ComputeIKSpotMicro(i, xyz ) 
        dq_des = np.zeros(3)
        tau_joint = kps*(q_des-q) + kds*(dq_des-dq)

        action[3*i:3*i+3] += tau_joint

    return action

  def spherical2cartesian(self,r,theta,psi):
    x = r*np.sin(theta)*np.cos(psi)
    y = r*np.sin(theta)*np.sin(psi)
    z = r*np.cos(theta)
    return np.array([x,y,z])


  def ScaleActionToTorqueRange(self,actions):
    """ Scale action form [-1,1] to full. """
    low_lim = -self._robot_config.TORQUE_LIMITS
    upp_lim =  self._robot_config.TORQUE_LIMITS
    u = np.clip(actions,-1,1)
    u = self._scale_helper(u, low_lim, upp_lim) 
    return u

  def ScaleActionToPosRange(self,actions):
    """ Scale action form [-1,1] to full. """
    low_lim = self._robot_config.LOWER_ANGLE_JOINT
    upp_lim = self._robot_config.UPPER_ANGLE_JOINT
    u = np.clip(actions,-1,1)
    u = self._scale_helper(u, low_lim, upp_lim) 
    u = u + self._robot_config.JOINT_OFFSETS
    #u = low_lim + 0.5 * (u + 1) * ( upp_lim - low_lim )  
    #u = np.clip(u,low_lim,upp_lim)
    #u = np.multiply(u, self._motor_direction)
    return u

  def ScaleActionToVelRange(self,actions):
    """ Scale action form [-1,1] to full. """
    low_lim = -self._robot_config.VELOCITY_LIMITS
    upp_lim =  self._robot_config.VELOCITY_LIMITS
    u = np.clip(actions,-1,1)
    u = self._scale_helper(u, low_lim, upp_lim) 
    return u

  def ScaleActionToIKRange(self,actions):
    """ Scale action from [-1,1] to desired end effector locations,
    and then corresponding joint positions. """
    low_lim = self._robot_config.IK_POS_LOW
    upp_lim = self._robot_config.IK_POS_UPP
    u = np.clip(actions,-1,1)
    u = self._scale_helper(u, low_lim, upp_lim) 
    # calculate joint positions
    jointPoses = np.zeros(12)
    for i in range(4):
      targetPos = u[3*i:3*i+3] 
      #print('compute IK')
      jointPoses[3*i:3*i+3] = self._ComputeIK(i,targetPos) 
    # clip by joint ranges (in case)
    jointPoses = np.clip(jointPoses, 
                          self._robot_config.LOWER_ANGLE_JOINT, 
                          self._robot_config.UPPER_ANGLE_JOINT)
    return jointPoses


  def ScaleActionDiagImpedanceControl(self,actions,use_force_if_contact_only=True):
    """ Scale action form [-1,1] to everything needed for impedance control. FOR TROTTING
    Assumes the following actions per control mode:

    should each foot get to choose its own force?

    IMPEDANCE:
      -2 * (x,y,z) pos; 2 x normal force (8 values)
    
    """
    # clip to make sure all actions in [-1,1]
    u = np.clip(actions,-1,1)

    # kp Cartesian and kdCartesian should in config file now
    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian

    # action is FR xyz, FL xyz, FR Fn, FL Fn
    scale_xyz = np.array([0.2,0.05,0.1]) # these are max ofsets
    FR_xyz_offset = scale_xyz * u[0:3]
    FL_xyz_offset = scale_xyz * u[3:6]
    imp_pos = self._robot_config.NOMINAL_IMPEDANCE_POS
    u_pos = imp_pos + np.concatenate(( FR_xyz_offset, FL_xyz_offset, FL_xyz_offset, FR_xyz_offset ))
    #print('nomin', imp_pos)
    #print('u_pos', u_pos)
    if len(u) < 8:
      u_force = np.zeros(4)
    elif len(u) == 8: 
      # feedforward forces 
      u_force = np.array([u[6], u[7],u[7],u[6]])

    elif len(u) == 10: 
      # feedforward forces 
      u_force = u[6:]

    else:
      raise ValueError('not implemented impedance control with this action space')

    if len(u) >= 8:
      low_lim_Fn = self._robot_config.MIN_NORMAL_FORCE
      upp_lim_Fn = self._robot_config.MAX_NORMAL_FORCE
      u_force = self._scale_helper(u_force, low_lim_Fn, upp_lim_Fn) 
      u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])
    else:
      u_force = np.zeros(12)
    # des e.e. vel
    u_vel = np.zeros(12)

    # kpCartesianArr = [kpCartesian]*4
    # # adjust feedforward force only if in contact
    # if use_force_if_contact_only:
    #   _,_,_,feetInContactBool = self.GetContactInfo()
    #   for i,inContact in enumerate(feetInContactBool):
    #     if inContact == 0: # not in contact
    #       u_force[3*i:3*i+3] = np.zeros(3)

    return self._ComputeFullImpedanceControlActions(u_pos,u_vel,u_force)

  def ScaleActionToImpedanceControl(self,actions,use_force_if_contact_only=True, velFlag=False):
    """ Scale action form [-1,1] to everything needed for impedance control. 
    Assumes the following actions per control mode:

    IMPEDANCE_POS: (REALLY IK)
      -4 * (x,y,z) pos(12 values), no force
    IMPEDANCE:
      -4 * (x,y,z) pos; 4 x normal force (16 values)
    [removed] IMPEDANCE_FORCE:
      -4 * normal force 
    IMPEDANCE_GAINS:
      -4 * (x,y,z) pos; 4 x Fn; kpCartesian; kdCartesian (22 values)
    IMPEDANCE_3D_FORCE:
      -4 * (x,y,z) pos; 4 x 3D contact force (24 values)
    IMPEDANCE_FULL:
      -4 * (x,y,z) pos/vel; 4 x 3D contact force (36 values)

    messy... lots of repetition
    """
    # clip to make sure all actions in [-1,1]
    u = np.clip(actions,-1,1)

    # kp Cartesian and kdCartesian should in config file now
    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian

    # so much repetition, clean this up
    if velFlag:
      if len(u) == 24:
        # action is 12 positions, 12 velocities
        u_pos = u[:12]
        u_vel = u[12:]
        # scale des xyz 
        low_lim_ik = self._robot_config.IK_POS_LOW_LEG
        upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
        u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
        # scale e.e. vel
        low_lim_vel = -5*np.ones(12)
        upp_lim_vel =  5*np.ones(12)
        u_vel = self._scale_helper(u_vel, low_lim_vel, upp_lim_vel) 
        # no feedforward force
        u_force = np.zeros(12)
      else:
        raise ValueError('not implemented impedance control with this action space')
    elif len(u) == 12: 
      # action is 12 positions 
      u_pos = u
      # scale des xyz 
      low_lim_ik = self._robot_config.IK_POS_LOW_LEG
      upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
      u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
      # feedforward forces are 0
      u_force = np.zeros(12) 
      # des e.e. vel
      u_vel = np.zeros(12)

    elif len(u) == 16: 
      # action is 12 positions + 4 normal forces
      u_pos = u[:12]
      u_force = u[12:]
      # scale des xyz 
      low_lim_ik = self._robot_config.IK_POS_LOW_LEG
      upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
      u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
      # scale des normal forces 
      low_lim_Fn = self._robot_config.MIN_NORMAL_FORCE
      upp_lim_Fn = self._robot_config.MAX_NORMAL_FORCE
      u_force = self._scale_helper(u_force, low_lim_Fn, upp_lim_Fn) 
      #u_force = [ np.array([0.,0.,fn]) for fn in u_force ]
      #u_force = np.concatenate( [R.T @ ffwd for ffwd in u_force])
      #u_force = np.concatenate( [R.T @ np.array([0.,0.,fn]) for fn in u_force])
      u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])
      # des e.e. vel
      u_vel = np.zeros(12)

    elif len(u) == 22: 
      # action is 12 positions, 4 Fn, 3 kpCartesian, 3 kdCartesian
      u_pos = u[:12]
      u_force = u[12:16]
      kpCartesian = u[16:19]
      kdCartesian = u[19:]
      # scale des xyz 
      low_lim_ik = self._robot_config.IK_POS_LOW_LEG
      upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
      u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
      # scale gains
      kpCartesian = np.diag(self._scale_helper(kpCartesian, 100, 800))
      kdCartesian = np.diag(self._scale_helper(kdCartesian,  10,  30))
      # scale des normal forces 
      low_lim_Fn = self._robot_config.MIN_NORMAL_FORCE
      upp_lim_Fn = self._robot_config.MAX_NORMAL_FORCE
      u_force = self._scale_helper(u_force, low_lim_Fn, upp_lim_Fn) 
      u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])
      # des e.e. vel
      u_vel = np.zeros(12)

    elif len(u) == 24: 
      # action is 12 positions, 4 3D contact forces
      u_pos = u[:12]
      u_force = u[12:]
      # scale des xyz 
      low_lim_ik = self._robot_config.IK_POS_LOW_LEG
      upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
      u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
      # scale 3D contact force
      low_lim_F = self._robot_config.MIN_NORMAL_FORCE_3D
      upp_lim_F = self._robot_config.MAX_NORMAL_FORCE_3D
      u_force = self._scale_helper(u_force, low_lim_F, upp_lim_F) 
      # des e.e. vel
      u_vel = np.zeros(12)

    elif len(u) == 36: 
      # action is 12 positions, 12 velocities, 4 3D contact forces
      low_lim = self._robot_config.FULL_IMPEDANCE_LOW
      upp_lim = self._robot_config.FULL_IMPEDANCE_UPP
      u = self._scale_helper(u, low_lim, upp_lim) 
      u_pos = u[:12]
      u_vel = u[12:24]
      u_force = u[24:]

    else:
      raise ValueError('not implemented impedance control with this action space')


    # build kpCartesianArr, apply kpCartesian only if not in contact with ground, like in MPC
    #kpCartesian = np.diag([500,500,100])
    #kdCartesian = np.diag([30,30,30])
    kpCartesianArr = [kpCartesian]*4

    # adjust feedforward force only if in contact
    if use_force_if_contact_only:
      _,_,_,feetInContactBool = self.GetContactInfo()

      """
      QUICK TEST
      """
      # for i in range(4):
      #   if np.linalg.norm(u_force[3*i:3*i+3]) != 0:
      #     kpCartesianArr[i] = np.zeros((3,3))

      for i,inContact in enumerate(feetInContactBool):
        if inContact == 0: # not in contact
          u_force[3*i:3*i+3] = np.zeros(3)

        # else:
        #   # in contact
        #   kpCartesianArr[i] = np.zeros((3,3))
        #   u_vel[3*i:3*i+3] = np.zeros(3)

    # Save for plotting (only if rendering)
    if self.render:
      self.save_xyz.extend(u_pos[:3])
      #self.save_Fn.append(u_force[2])
      self.save_Fn.append(u_force[2]) # should be approx. correct when standing
      numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = self.GetContactInfo()
      self.save_real_Fn.append(feetNormalForces[0])
      _, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(0)
      self.save_real_xyz.extend(ee_pos_legFrame)


    # desired foot velocity (may want to change)
    #vd = np.zeros(3)
    # print('pos', u_pos)
    # print('vel', u_vel)
    # print('ffwd', u_force)
    # print('kpCa', [kp.diagonal() for kp in kpCartesianArr])
    self._des_foot_pos = u_pos.copy()

    # find corresponding torques
    action = np.zeros(12)
    # body pos, orientation matrix, velocity
    #basePos = np.asarray(self.GetBasePosition())
    R = self.GetBaseOrientationMatrix()

    # save for ROS
    # feedforwardForce = np.zeros(12)
    # pDes = np.zeros(12)
    # vDes = np.zeros(12)
    # kpCartesianDiag = np.zeros(12)
    # kdCartesianDiag = np.zeros(12)

    for i in range(4):
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
      # desired foot position
      Pd = u_pos[i*3:i*3+3]
      # desired foot velocity
      vd = u_vel[i*3:i*3+3]
      # desired force (rotated into body frame)
      if len(u) <= 16:
        # learning just desired normal force in world plane
        Fd = R.T @ u_force[i*3:i*3+3]
      else:
        # learn 3d force, makes more sense in leg frame, since contact surface may not be flat
        Fd = u_force[i*3:i*3+3] 
      # foot velocity in leg frame
      foot_linvel = J @ self.GetMotorVelocities()[i*3:i*3+3]
      # calculate torque
      #tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
      #print(kpCartesianArr[i])
      tau = J.T @ ( Fd + kpCartesianArr[i] @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
      action[i*3:i*3+3] = tau

      # save info
      # feedforwardForce[i*3:i*3+3] = Fd
      # pDes[i*3:i*3+3] = Pd
      # vDes[i*3:i*3+3] = vd
      # kpCartesianDiag[i*3:i*3+3] = kpCartesianArr[i].diagonal()
      # kdCartesianDiag[i*3:i*3+3] = kdCartesian.diagonal()
      #print('ee pos leg ', i, ee_pos_legFrame, 'pybullet', self.GetLocalEEPos(i))

    #print(action)
    # for ROS
    info = {}
    # info['feedforwardForce'] = feedforwardForce
    # info['pDes'] = pDes
    # info['vDes'] = vDes
    # info['kpCartesianDiag'] = kpCartesianDiag
    # info['kdCartesianDiag'] = kdCartesianDiag

    return action, info

  def _ComputeFullImpedanceControlActions(self,u_pos,u_vel,u_force):
    """Get torques for specific forces and actions. 
    Note forces coming from ROS are going to be Vec3 that are already rotated in local frame.
    This function only for 1 leg
    """
    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian
    #kdCartesian = np.diag([30,30,30])
    # kpCartesianArr
    kpCartesianArr = [kpCartesian] * 4 
    # rotation, joint velocities
    R = self.GetBaseOrientationMatrix()
    qd = self.GetMotorVelocities()
    action = np.zeros(12)

    # u_pos = np.array([0,-0.08,-0.3,
    #                   0, 0.08,-0.3,
    #                   0,-0.08,-0.3,
    #                   0, 0.08,-0.3])
    #u_force = np.zeros(12)
    # zero force if in air
    _,_,_,feetInContactBool = self.GetContactInfo()
    for i,inContact in enumerate(feetInContactBool):
      if inContact == 0:
        u_force[3*i:3*i+3] = np.zeros(3)

    for i in range(4):
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
      # desired foot position
      Pd = u_pos[i*3:i*3+3]
      # desired foot velocity
      vd = u_vel[i*3:i*3+3]
      # desired force (rotated into body frame)
      #Fd = R.T @ np.array([0.,0.,u_force[i]])
      Fd = R.T @ u_force[i*3:i*3+3] # 
      #Fd = R @ u_force[i*3:i*3+3] # taking a world frame force into body frame
      # foot velocity in leg frame
      foot_linvel = J @ qd[i*3:i*3+3]
      # calculate torque
      #tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
      #print(kpCartesianArr[i])
      tau = J.T @ ( Fd + kpCartesianArr[i] @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
      action[i*3:i*3+3] = tau

    return action


  def GetXYZFn(self):
    return self.save_xyz, self.save_Fn

  def _ComputeLegImpedanceControlActions(self,u_pos,u_vel,kpCartesian,kdCartesian,u_force,legID):
    """Get torques for specific forces and actions. 
    Note forces coming from ROS are going to be Vec3 that are already rotated in local frame.
    This function only for 1 leg
    """
    # rotation matrix
    #R = self.GetBaseOrientationMatrix()
    # Jacobian and current foot position in hip frame
    J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(legID)
    # desired foot position
    Pd = u_pos 
    # desired foot velocity 
    vd = u_vel 
    # desired force (rotated)
    Fd = u_force 
    # foot velocity in leg frame
    foot_linvel = J @ self.GetMotorVelocities()[legID*3:legID*3+3]
    # calculate torque
    tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
    return tau

  def _ComputeJacobianAndPositionUSC(self, legID, specific_delta_q=None):
    """Translated to python/pybullet from: 
    https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/LegController.cpp
    can also use pybullet function calculateJacobian()
    IN LEG FRAME
    (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
    # from LegController:
    # * Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
    """
    if specific_delta_q is not None:
      # used for a specific contribution, may not be actual state, just to get Jacobian
      q = specific_delta_q
    else:
      # joint positions of leg
      q = self.GetMotorAngles()[legID*3:legID*3+3]
      # test for Laikago, will be 0 for other robots (b/c added offset in GetMotorAngles() - why?)
      q -= self._robot_config.JOINT_OFFSETS[legID*3:legID*3+3]

    # Foot position in base frame
    #linkPosInBase = self.GetLocalEEPos(legID)
    # local position of E.E. in leg frame (relative to hip)
    #pos = np.array(linkPosInBase) - HIP_POSITIONS[legID]
    pos = np.zeros(3)

    l1 = self._robot_config.HIP_LINK_LENGTH
    l2 = self._robot_config.THIGH_LINK_LENGTH
    l3 = self._robot_config.CALF_LINK_LENGTH

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

  def _ComputeInverseKinematicsUSC(self,leg,pDes):
    """ From LASER repo. """
    l1 = self._robot_config.HIP_LINK_LENGTH
    l2 = self._robot_config.THIGH_LINK_LENGTH
    l3 = self._robot_config.CALF_LINK_LENGTH

    x = pDes[0]
    y = pDes[1]
    z = pDes[2]

    z_1 = np.sqrt(y*y + z*z - l1*l1)
    theta_r = np.arctan2(x, z_1)

    q0 = 2*np.arctan((l1 - np.sqrt(-z*z + z_1*z_1 + l1*l1))/(np.abs(z) + z_1))
    q1 = np.arccos((l2*l2 + x*x + z_1*z_1 -l3*l3)/(2*l2*np.sqrt(z_1*z_1 + x*x)))+ theta_r
    q2 = np.arccos((x*x + y*y + z*z -l1*l1 - l2*l2 - l3*l3)/(2*l2*l3))

    qDes = np.zeros(3)
    if leg==0 or leg ==2:
      qDes[0] = -q0
      qDes[1] = q1
      qDes[2] = -q2 

    if leg == 1 or leg == 3:
      qDes[0] = q0
      qDes[1] = q1
      qDes[2] = -q2

    return qDes


  def _ComputeIKSpotMicro(self,legID, xyz_coord):
    """ From SpotMicro: 
    https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
    """
    # rename 
    shoulder_length = self._robot_config.HIP_LINK_LENGTH
    elbow_length = self._robot_config.THIGH_LINK_LENGTH
    wrist_length =  self._robot_config.CALF_LINK_LENGTH
    # coords
    x = xyz_coord[0]
    y = xyz_coord[1]
    z = xyz_coord[2]

    # get_domain
    D = (y**2 + (-z)**2 - shoulder_length**2 +
        (-x)**2 - elbow_length**2 - wrist_length**2) / (
                 2 * wrist_length * elbow_length)

    D = np.clip(D, -1.0, 1.0)

    # check Right vs Left leg for hip angle
    sideSign = 1
    if legID == 0 or legID == 2:
      sideSign = -1

    # Right Leg Inverse Kinematics Solver
    wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
    sqrt_component = y**2 + (-z)**2 - shoulder_length**2
    if sqrt_component < 0.0:
        # print("NEGATIVE SQRT")
        sqrt_component = 0.0
    shoulder_angle = -np.arctan2(z, y) - np.arctan2(
        np.sqrt(sqrt_component), sideSign*shoulder_length)
    elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
        wrist_length * np.sin(wrist_angle),
        elbow_length + wrist_length * np.cos(wrist_angle))
    joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
    return joint_angles

  

  def _ComputeIK(self,leg_id,targetPos):
    """ Calculate IK for legid, with local target position.

    TODO: verify how many iterations of IK needed, as well as residual threshold.
    See Aliengo/pybullet_tests/aliengo_ik_tests.py
    NOTE: arguments to calculateInverseKinematics MUST be lists (numpy arrays will seg fault)
    """
    # need current joint states
    currJointPos = list(self.GetMotorAngles())
    ee_index = self._foot_link_ids[leg_id]
    # print('do IK')
    # print('ffoot link ids', self._foot_link_ids)
    # print('ee_index', ee_index)
    # print(self._robot_config.LL)
    # print(self._robot_config.UL, self._robot_config.JR,self._robot_config.RP,currJointPos)
    # targetPos = [0.3,-0.11,0.4]
    jointPosesCurr = self._pybullet_client.calculateInverseKinematics(self.quadruped, 
                        ee_index, 
                        list(targetPos), 
                        lowerLimits=self._robot_config.LL, 
                        upperLimits=self._robot_config.UL,
                        jointRanges=self._robot_config.JR, 
                        restPoses=self._robot_config.RP,
                        solver=1,
                        currentPositions=currJointPos,
                        residualThreshold=1e-4,
                        maxNumIterations=20) # 200

    #print(jointPosesCurr)
    return jointPosesCurr[3*leg_id:3*leg_id+3]
  ######################################################################################
  # RESET related
  ######################################################################################
  def Reset(self, reload_urdf=False):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
    """
    if reload_urdf:
      self._LoadRobotURDF()
      self._BuildJointNameToIdDict()
      self._BuildUrdfIds()
      self._RemoveDefaultJointDamping()
      self._SetLateralFriction()
      self._BuildMotorIdList()
      self._RecordMassAndInertiaInfoFromURDF()
      self._SetMaxJointVelocities()

      self.ResetPose(add_constraint=True)
      # self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, 
      #                                                       self._GetDefaultInitPosition(),
      #                                                       self._GetDefaultInitOrientation())
      if self._on_rack:
        # self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
        #                                        self._pybullet_client.JOINT_FIXED, [0, 0, 0],
        #                                        [0, 0, 0], [0, 0, 1])
        # check orientation 
        self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
                                               self._pybullet_client.JOINT_FIXED, [0, 0, 0],
                                               [0, 0, 0], [0, 0, 1], childFrameOrientation=self._GetDefaultInitOrientation())
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, 
                                                            self._GetDefaultInitPosition(),
                                                            self._GetDefaultInitOrientation())
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors
    self._des_foot_pos = self._robot_config.NOMINAL_IMPEDANCE_POS
    """ TO ADD:
    self._observation_history.clear()
    self._step_counter = 0
    self._state_action_counter = 0
    self._is_safe = True
    self._last_action = None

    self._SettleDownForReset(default_motor_angles, reset_time)

    if self._enable_action_filter:
      self._ResetActionFilter()

    return
    """

  def ResetPose(self, add_constraint):
    """From laikago.py """
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    # for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
    #   if "hip_motor_2_chassis_joint" in name:
    #     angle = INIT_MOTOR_ANGLES[i] #+ HIP_JOINT_OFFSET
    #   elif "upper_leg_2_hip_motor_joint" in name:
    #     angle = INIT_MOTOR_ANGLES[i] #+ UPPER_LEG_JOINT_OFFSET
    #   elif "lower_leg_2_upper_leg_joint" in name:
    #     angle = INIT_MOTOR_ANGLES[i] #+ KNEE_JOINT_OFFSET
    #   else:
    #     raise ValueError("The name %s is not recognized as a motor joint." %
    #                      name)
    #   self._pybullet_client.resetJointState(
    #       self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

    for i, jointId in enumerate(self._joint_ids):
      angle = self._robot_config.INIT_MOTOR_ANGLES[i] + self._robot_config.JOINT_OFFSETS[i]
      self._pybullet_client.resetJointState(
          self.quadruped, jointId, angle, targetVelocity=0)

  ######################################################################################
  # [EXPERIMENTAL] Careful with these... all a bit hacky, frequently commenting/changing things
  ######################################################################################
  def SetPoseAndSettle(self,basePosOrn,joint_states, settle_threshold=0.01):
    """ Set robot base pos, joint states, and simulate until velocities are under some threshold. """
    base_pos, base_orn = np.split(basePosOrn, 2)
    self.ResetBasePositionAndOrientation(base_pos,base_orn)
    joint_pos, joint_vel = self.SetJointStates(joint_states)
    # set joints with PD control
    real_control_mode = self._motor_control_mode
    self._motor_control_mode = "PD"
    self.ApplyAction(joint_pos)
    # switch back to whatever mode we were really in
    self._motor_control_mode = real_control_mode
    #print('\nin control mode', self._motor_control_mode, '\n')
    #self._settle(settle_threshold)

    # make sure the base position and orientation is close to what we wanted!
    actual_basePosOrn = self.GetBaseState()[0:6]
    #print(basePosOrn,actual_basePosOrn)
    assert np.linalg.norm(basePosOrn - actual_basePosOrn) < 0.1
    #self._disable_motor()
  
  def ResetBasePositionAndOrientation(self, base_pos, base_orn ):
    """ Resets base position and orientation. Orientation can be Euler or quaternion """
    if len(base_orn) == 3: #Euler, but need quaternion
      base_orn = self._pybullet_client.getQuaternionFromEuler(base_orn)
    self._pybullet_client.resetBasePositionAndOrientation(self.quadruped,base_pos,base_orn)

  def ResetBaseVelocity(self, base_linvel, base_angvel ):
    """ Resets base linear and angular velocities. """
    self._pybullet_client.resetBaseVelocity(self.quadruped,base_linvel, base_angvel)

  def SetJointStates(self,joint_states):
    """Take in joint positions and velocities, and set joint states. """
    joint_pos, joint_vel = np.split(joint_states, 2)
    for i,jointId in enumerate(self._joint_ids):
      self._pybullet_client.resetJointState(self.quadruped, jointId, joint_pos[i], targetVelocity=joint_vel[i])
    return joint_pos, joint_vel

  def _settle(self,settle_threshold=0.2, max_settle_sim_time=5):
    """ Simulate until robot overall velocity falls below a threshold. Call after setting pose. """
    settle_sim_cntr = 0
    while True:
      print('settling')
      self._pybullet_client.stepSimulation()
      all_vels = [self.GetBaseLinearVelocity(), self.GetBaseAngularVelocity(), self.GetMotorVelocities()]
      total_vel_scalar = np.sum( [np.linalg.norm(i) for i in all_vels] )
      if total_vel_scalar < settle_threshold:
        break
      if settle_sim_cntr * self.time_step > max_settle_sim_time:
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
      self._pybullet_client.resetJointState(self.quadruped, self._joint_ids[i], jointPos, targetVelocity=jointVels[i])

    if self.render:
      #self._p.stepSimulation()
      time.sleep(max(self.time_step,0.01))
  ######################################################################################
  # URDF related
  ######################################################################################
  def _LoadRobotURDF(self):
    """Loads the URDF file for the robot."""
    urdf_file =  os.path.join(self._urdf_root, self._robot_config.URDF_FILENAME)
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          urdf_file,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          urdf_file, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1] # just base link
    self._leg_link_ids = []   # all leg links (hip, thigh, calf)
    self._motor_link_ids = [] # all leg links (hip, thigh, calf)

    self._joint_ids=[]      # all motor joints
    self._hip_ids = []      # hip joint indices only
    self._thigh_ids = []    # thigh joint indices only
    self._calf_ids = []     # calf joint indices only
    self._foot_link_ids = [] # foot joint indices

    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      # print(joint_info)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if self._robot_config._CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_ids = [joint_id] #.append(joint_id) # _should_ be only one!
      elif self._robot_config._HIP_NAME_PATTERN.match(joint_name):
        self._hip_ids.append(joint_id)
      elif self._robot_config._THIGH_NAME_PATTERN.match(joint_name):
        self._thigh_ids.append(joint_id)
      elif self._robot_config._CALF_NAME_PATTERN.match(joint_name):
        self._calf_ids.append(joint_id)
      elif self._robot_config._FOOT_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      else:
        continue
        raise ValueError("Unknown category of joint %s" % joint_name)


    # print('chassis',self._chassis_link_ids)
    # for i in range(-1,num_joints):
    #   print(self._pybullet_client.getDynamicsInfo(self.quadruped, i))

    # everything associated with the leg links
    self._joint_ids.extend(self._hip_ids)
    self._joint_ids.extend(self._thigh_ids)
    self._joint_ids.extend(self._calf_ids)
    # sort in case any weird order
    self._joint_ids.sort()
    self._hip_ids.sort()
    self._thigh_ids.sort()
    self._calf_ids.sort()
    self._foot_link_ids.sort()

    # print('joint ids', self._joint_ids)
    # sys.exit()
    return

  def _RecordMassAndInertiaInfoFromURDF(self):
    """Records the mass information from the URDF file.
    Divide into base, leg, foot, and total (all links)
    """
    # base 
    self._base_mass_urdf = []
    self._base_inertia_urdf = []
    # might update this one in future.. with masses at different parts of robot
    for chassis_id in self._chassis_link_ids:
      self._base_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
      self._base_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[2])

    # leg masses
    self._leg_masses_urdf = []
    self._leg_inertia_urdf = []
    for leg_id in self._joint_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
      self._leg_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[2])

    # foot masses
    self._foot_masses_urdf = []
    self._foot_inertia_urdf = []
    for foot_id in self._foot_link_ids:
      self._foot_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, foot_id)[0])
      self._foot_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, foot_id)[2])


    # all masses present in URDF
    self._total_mass_urdf = []
    self._total_inertia_urdf = []
    # don't forget base (which is at -1)
    for j in range(-1,self._pybullet_client.getNumJoints(self.quadruped)):
      self._total_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped,j)[0])
      self._total_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped,j)[2])

    # set originals as needed
    self._original_base_mass_urdf = tuple(self._base_mass_urdf)
    self._original_leg_masses_urdf = tuple(self._leg_masses_urdf)
    self._original_foot_masses_urdf = tuple(self._foot_masses_urdf)
    self._original_total_mass_urdf = tuple(self._total_mass_urdf)

    self._original_base_inertia_urdf = tuple(self._base_inertia_urdf)
    self._original_leg_inertia_urdf = tuple(self._leg_inertia_urdf)
    self._original_foot_inertia_urdf = tuple(self._foot_inertia_urdf)
    self._original_total_inertia_urdf = tuple(self._total_inertia_urdf)

    #print('total mass', sum(self._total_mass_urdf))

  def _BuildJointNameToIdDict(self):
    """_BuildJointNameToIdDict """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    # self._motor_id_list = [self._joint_name_to_id[motor_name] 
    #                         for motor_name in self._robot_config.MOTOR_NAMES]
    self._motor_id_list = self._joint_ids

  def _RemoveDefaultJointDamping(self):
    """Pybullet convention/necessity  """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._pybullet_client.changeDynamics(
          joint_info[0], -1, linearDamping=0, angularDamping=0)

  def _SetLateralFriction(self,lateral_friction=1.0):
    """ Lateral friction to be set for every link. 
    NOTE: pybullet default is 0.5 - so call to set to 1 as default
    (compared to Laikago urdf where it is set to 1).
    """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for i in range(-1,num_joints):
      self._pybullet_client.changeDynamics(self.quadruped, i, lateralFriction=lateral_friction)

  def _SetMaxJointVelocities(self):
    """Set maximum joint velocities from robot_config, the pybullet default is 100 rad/s """
    for i, link_id in enumerate(self._joint_ids):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, maxJointVelocity=self._robot_config.VELOCITY_LIMITS[i])  

  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetFootMassesFromURDF(self):
    """Get the mass of the feet from the URDF file."""
    return self._foot_masses_urdf

  def GetTotalMassFromURDF(self):
    """Get the total mass from all links in the URDF."""
    return self._total_mass_urdf

  def SetBaseMass(self, base_mass):
    """Set base mass, should be link [-1]. """
    self._pybullet_client.changeDynamics(self.quadruped, self._chassis_link_ids[0], mass=base_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs. Should be 12 values (hips, thighs, calves). """
    if len(leg_masses) != len(self._joint_ids):
      raise ValueError('Must set all leg masses.')
    for i, link_id in enumerate(self._joint_ids):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[i])

  def SetFootMasses(self, foot_masses):
    """Set the mass of the feet. Should be 4 values. """
    if len(foot_masses) != len(self._foot_link_ids):
      raise ValueError('Must set all foot masses.')
    for i, link_id in enumerate(self._foot_link_ids):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=foot_masses[i])

  def SetBaseInertia(self, base_inertia):
    """Set base inertia, should be a vec3 local inertia diagonal elements at link [-1]. """
    self._pybullet_client.changeDynamics(self.quadruped, self._chassis_link_ids[0], localInertiaDiagonal=base_inertia)

  def SetLegInertias(self, leg_inertias):
    """Set the inertias of each leg component. 
    Should be 12 vec3 local inertia diagonal elements (hips, thighs, calves). """
    if len(leg_inertias) != len(self._joint_ids):
      raise ValueError('Must set all leg inertias.')
    for i, link_id in enumerate(self._joint_ids):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, localInertiaDiagonal=leg_inertias[i])

  def SetFootInertias(self, foot_inertias):
    """Set the inertias of the feet. Should be 4 vec3 local inertia diagonal elements. """
    if len(foot_inertias) != len(self._foot_link_ids):
      raise ValueError('Must set all foot masses.')
    for i, link_id in enumerate(self._foot_link_ids):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, localInertiaDiagonal=foot_inertias[i])

  def RandomizePhysicalParams(self, mass_percent_change=0.1, inertia_percent_change=0.1, base_only=False):
    """Randomize physical robot parameters: masses and inertias.
    Args: % sample range of baseline for masses, inertias compared to original values
      i.e. for a mass of 2kg and mass_percent_change=0.1, the new mass will be in range [1.8,2.2]

      NOTE: inertia needs some care.. and does not work below. Other papers (minitaur/spot mini) 
        don't actually randomize the inertia. Is this done automatically when setting mass? 
    """
    # Mass
    base_mass = np.array(self._original_base_mass_urdf)
    self.SetBaseMass(self._rand_helper_1d(base_mass,mass_percent_change)[0])
    if not base_only:
      leg_masses = np.array(self._original_leg_masses_urdf)
      foot_masses = np.array(self._original_foot_masses_urdf)
      self.SetLegMasses(self._rand_helper_1d(leg_masses,mass_percent_change))
      self.SetFootMasses(self._rand_helper_1d(foot_masses,mass_percent_change))
      #print('masses', base_mass,leg_masses,foot_masses)

    # Inertia
    # base_inertia = self._original_base_inertia_urdf 
    # self.SetBaseInertia(self._rand_helper_2d(base_inertia,inertia_percent_change))
    # if not base_only:
    #   leg_inertias = self._original_leg_inertia_urdf 
    #   foot_inertias = self._original_foot_inertia_urdf
    #   #self._original_total_inertia_urdf
    #   self.SetLegInertias(self._rand_helper_2d(leg_inertias,inertia_percent_change))
    #   self.SetFootInertias(self._rand_helper_2d(foot_inertias,inertia_percent_change))

  def _rand_helper_1d(self,orig_vec,percent_change):
    """Scale appropriately random in low/upp range, 1d vector """
    vec = np.zeros(len(orig_vec))
    for i,elem in enumerate(orig_vec):
      delta = percent_change*orig_vec[i]
      # vec[i] = orig_vec[i] + ( 2*delta*np.random.random() - delta)
      vec[i] = orig_vec[i] + ( 2*delta*np.random.random() - delta)
    #print(orig_vec, vec)
    return vec

  def _rand_helper_2d(self,orig_vec,percent_change):
    """Scale appropriately random in low/upp range, 2d vector """
    new_vec = [] #np.zeros(len(orig_vec))
    for i,elem in enumerate(orig_vec):
      new_subelem = []
      for subelem in elem: 
        delta = percent_change*subelem #orig_vec[i]
        new_subelem.append( subelem + ( 2*delta*np.random.random() - delta) )
      new_vec.append(tuple(new_subelem))
    #print(orig_vec, new_vec)
    return new_vec
  ######################################################################################
  # Misc
  ######################################################################################
  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, lateralFriction=foot_friction)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)


  def ApplyExternalForce(self, force_vec):
    """ Apply external force at base. """
    self._pybullet_client.applyExternalForce(self.quadruped, 
                                            -1, # at base
                                            forceObj=list(force_vec),
                                            posObj=self.GetBasePosition(),
                                            flags=self._pybullet_client.WORLD_FRAME)




  def get_cam_view(self):
    original_cam_look_direction = np.array([1, 0, 0])  # Same as original robot orientation

    pos = self.GetBasePosition()
    orientation = self.GetBaseOrientation()
    axis, ori = self._pybullet_client.getAxisAngleFromQuaternion(orientation)
    axis = np.array(axis)

    original_cam_up_vector = np.array([0, 0, 1])  # Original camera up vector

    new_cam_up_vector = np.cos(ori) * original_cam_up_vector + np.sin(ori) * np.cross(axis, original_cam_up_vector) + (
            1 - np.cos(ori)) * np.dot(axis, original_cam_up_vector) * axis  # New camera up vector

    new_cam_look_direction = np.cos(ori) * original_cam_look_direction + np.sin(ori) * np.cross(axis, original_cam_look_direction) + (
            1 - np.cos(ori)) * np.dot(axis, original_cam_look_direction) * axis  # New camera look direction

    new_target_pos = pos + new_cam_look_direction  # New target position for camera to look at

    new_cam_pos = pos + 0.3 * new_cam_look_direction

    viewMatrix = self._pybullet_client.computeViewMatrix(
      cameraEyePosition=new_cam_pos,
      cameraTargetPosition=new_target_pos,
      cameraUpVector=new_cam_up_vector)

    projectionMatrix = self._pybullet_client.computeProjectionMatrixFOV(
      fov=45.0,
      aspect=1.0,
      nearVal=0.01,
      farVal=10)

    width, height, rgbImg, depthImg, segImg = self._pybullet_client.getCameraImage(
      width=400,
      height=300,
      viewMatrix=viewMatrix,
      projectionMatrix=projectionMatrix)

    return rgbImg, depthImg