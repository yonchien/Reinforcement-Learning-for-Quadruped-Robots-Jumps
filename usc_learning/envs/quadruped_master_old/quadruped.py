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
from usc_learning.envs.quadruped_master_old import quadruped_motor
from usc_learning.utils.robot_utils import FormatFRtoAll

""" minitaur """
# overheat protection?
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
BASE_LINK_ID = -1

# general
CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD", "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE" , "IMPEDANCE_FORCE"]

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
    #self._max_force = 3.5
    self._accurate_motor_model_enabled = accurate_motor_model_enabled

    # motor control mode for accurate motor model, should only be torque or position at this low level
    if motor_control_mode in ["PD","IK"]:
      self._motor_control_mode = "PD"
    elif motor_control_mode in ["TORQUE","FORCE","IMPEDANCE","IMPEDANCE_FORCE"]:
      self._motor_control_mode = "TORQUE"
    else:
      raise ValueError('motor control mode unclear')

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
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self.Reset(reload_urdf=True)

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
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
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


  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

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
    observation.extend(self.GetMotorTorques().tolist())
    observation.extend(list(self.GetBaseOrientation()))
    return observation


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

  def GetObservationUpperBound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.array([0.0] * self.GetObservationDimension())
    upper_bound[0:self.num_motors] = math.pi  # Joint angle.
    upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
    upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    return upper_bound

  def GetObservationLowerBound(self):
    """Get the lower bound of the observation."""
    return -self.GetObservationUpperBound()

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
    upper_bound[3 * self.num_motors+4+3:3 * self.num_motors+4+6] = (4.,4.,4.) # base lin vel max
    upper_bound[3 * self.num_motors+4+6:3 * self.num_motors+4+9] = (4.,4.,4.) # base ang vel max
    return upper_bound

  def GetExtendedObservationLowerBound(self):
    """Get the lower bound of the observation."""
    lower_bound = np.array([0.0] * len(self.GetExtendedObservation()))
    lower_bound[0:self.num_motors] = (self._robot_config.LOWER_ANGLE_JOINT)  # Joint angle.
    lower_bound[self.num_motors:2 * self.num_motors] = (-self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    lower_bound[2 * self.num_motors:3 * self.num_motors] = (-self._robot_config.TORQUE_LIMITS)  # Joint torque.
    lower_bound[3 * self.num_motors:3 * self.num_motors+4] = -1.0  # Quaternion of base orientation.
    lower_bound[3 * self.num_motors+4:3 * self.num_motors+4+3] = (-2.,-2.,0.1) # base pos min
    lower_bound[3 * self.num_motors+4+3:3 * self.num_motors+4+3+3] = (-4.,-4.,-4.) # base lin vel min
    lower_bound[3 * self.num_motors+4+6:3 * self.num_motors+4+9] = (-4.,-4.,-4.) # base ang vel min
    return lower_bound

  def GetObservationDimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self.GetObservation())

  ######################################################################################
  # INPUTS: set torques, ApplyAction, etc.
  ######################################################################################
  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.POSITION_CONTROL,
                                                targetPosition=desired_angle,
                                                positionGain=self._kp,
                                                velocityGain=self._kd,
                                                force=self._max_force)

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

      
      if self._motor_control_mode == "PD":
        """ clamp motor commands by the joint limits, in case weird things happen """
        max_angle_change = self._robot_config.MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        motor_commands = np.clip(motor_commands,
                               q - max_angle_change,
                               q + max_angle_change)
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
      raise ValueError("Should be using accurate motor model.")
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
    """Helper to linearly scale from [-1,1] to lower/upper limits. """
    new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
    # verify clip
    new_a = np.clip(new_a, lower_lim, upper_lim)
    return new_a

  def ScaleActionToTorqueRange(self,actions):
    """ Scale action form [-1,1] to full. """
    upp_lim = -self._robot_config.TORQUE_LIMITS
    low_lim =  self._robot_config.TORQUE_LIMITS
    u = np.clip(actions,-1,1)
    u = self._scale_helper(u, low_lim, upp_lim) 
    return u

  def ScaleActionToPosRange(self,actions):
    """ Scale action form [-1,1] to full. """
    upp_lim = self._robot_config.UPPER_ANGLE_JOINT
    low_lim = self._robot_config.LOWER_ANGLE_JOINT
    u = np.clip(actions,-1,1)
    u = low_lim + 0.5 * (u + 1) * ( upp_lim - low_lim )  
    u = np.clip(u,low_lim,upp_lim)
    #u = np.multiply(u, self._motor_direction)
    return u

  def ScaleActionToImpedanceControl(self,actions):
    """ Scale action form [-1,1] to everything needed for impedance control. 
    Assumes the following action order for each leg:
    -4 x (x,y,z) pos; 4 x normal force
    -16 values

    should this also output desired foot velocities?
    """
    # clip to make sure all actions in [-1,1]

    # TODO:
    # should only scale IF is an RL action, otherwise no
    # or: different function?
    u = np.clip(actions,-1,1)
    if len(u) == 4: # action is 12 positions + 4 normal forces
      # use feet positions
      hip_offset = self._robot_config.hip_offset #0.083
      zd_hip = 0.4
      Pd = np.array([0,-hip_offset,-zd_hip])
      u_pos = np.concatenate(FormatFRtoAll(Pd))
      u_force = u
      #print('\nright\n')
    elif len(u) == 16: # action is 12 positions + 4 normal forces
      u_pos = u[:12]
      u_force = u[12:]
      # scale des xyz 
      low_lim_ik = self._robot_config.IK_POS_LOW
      upp_lim_ik = self._robot_config.IK_POS_UPP
      u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
    else:
      raise ValueError('not implemented impedance control with this action space')

    # scale des normal forces 
    low_lim_Fn = self._robot_config.MIN_NORMAL_FORCE
    upp_lim_Fn = self._robot_config.MAX_NORMAL_FORCE
    u_force = self._scale_helper(u_force, low_lim_Fn, upp_lim_Fn) 

    """
    TODO:
    clip the max change in end effector position and normal force delta (what is reasonable)?
    clip normal force delta between time steps? how much?
    -edit: this seems to perform MUCH worse!
    """
    # get current foot locations
    #print(u_pos)
    # curr_ee = np.concatenate([self.GetLocalEEPos(i) for i in range(4)])
    # MAX_POS_CHANGE_PER_STEP = 0.05
    # u_pos = np.clip(u_pos, 
    #                 curr_ee - MAX_POS_CHANGE_PER_STEP, 
    #                 curr_ee + MAX_POS_CHANGE_PER_STEP)
    #time.sleep(0.2)
    #print('curr_ee',curr_ee)
    #print('u_pos', u_pos)
    #u_force = -55*np.ones(4)

    # kp and kd cartesian, should really be in config file?
    kpCartesian = np.diag([800,1400,1400])
    kdCartesian = np.diag([30,30,30])

    kpCartesian = np.diag([500,500,500])
    kdCartesian = np.diag([10,10,10])
    # desired foot velocity (is this true though?)
    vd = np.zeros(3)

    # find corresponding torques
    action = np.zeros(12)
    # body pos, orientation matrix, velocity
    basePos = np.asarray(self.GetBasePosition())
    R = self.GetBaseOrientationMatrix()
    for i in range(4):
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
      # desired foot position
      Pd = u_pos[i*3:i*3+3]
      # desired force (rotated)
      Fd = R.T @ np.array([0.,0.,u_force[i]])
      # foot velocity in leg frame
      foot_linvel = J @ self.GetMotorVelocities()[i*3:i*3+3]
      # calculate torque
      tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
      action[i*3:i*3+3] = tau

    return action


  def _ComputeJacobianAndPositionUSC(self, legID):
    """Translated to python/pybullet from: 
    https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/LegController.cpp
    can also use pybullet function calculateJacobian()
    IN LEG FRAME
    (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
    # from LegController:
    # * Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
    """

    # joint positions of leg
    q = self.GetMotorAngles()[legID*3:legID*3+3]
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
      """ 
      TO ADD:
      _RecordInertiaInfoFromURDF
      """
      self._BuildMotorIdList()
      self._RecordMassAndInertiaInfoFromURDF()
      #print("\n", self.GetBasePosition(), "\n")
      self.ResetPose(add_constraint=True)
      if self._on_rack:
        self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
                                               self._pybullet_client.JOINT_FIXED, [0, 0, 0],
                                               [0, 0, 0], [0, 0, 1])
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, 
                                                            self._GetDefaultInitPosition(),
                                                            self._GetDefaultInitOrientation())
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors

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
  # Careful with these... all a bit hacky, frequently commenting/changing things
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
      #print(joint_info)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if self._robot_config._CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_id = [joint_id] #.append(joint_id) # _should_ be only one!
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
    # sys.exit()
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

  def _BuildJointNameToIdDict(self):
    """_BuildJointNameToIdDict """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    self._motor_id_list = [self._joint_name_to_id[motor_name] 
                            for motor_name in self._robot_config.MOTOR_NAMES]

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


  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetTotalMassFromURDF(self):
    """Get the total mass from all links in the URDF."""
    return self._total_mass_urdf

  def SetBaseMass(self, base_mass):
    self._pybullet_client.changeDynamics(self.quadruped, BASE_LINK_ID, mass=base_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. All four leg_links have the same mass,
    which is leg_masses[0]. All four motors have the same mass, which is
    leg_mass[1].

    Args:
      leg_masses: The leg masses. leg_masses[0] is the mass of the leg link.
        leg_masses[1] is the mass of the motor.
    """
    raise NotImplementedError("Not debugged, leftover from minitaur. ")
    for link_id in LEG_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[0])
    for link_id in MOTOR_LINK_ID:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[1])

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
