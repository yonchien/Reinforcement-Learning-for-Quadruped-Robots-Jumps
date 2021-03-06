# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pybullet simulation of a Laikago robot."""
import math
import os
import re
import numpy as np
import pybullet as pyb

from usc_learning.envs.google_robots import laikago_pose_utils
from usc_learning.envs.google_robots import laikago_constants
from usc_learning.envs.google_robots import laikago_motor
from usc_learning.envs.google_robots import minitaur
from usc_learning.envs.google_robots import robot_config
from usc_learning.envs.google_robots import locomotion_gym_config

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.array([-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = -0.6
KNEE_JOINT_OFFSET = 0.66
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 0.3
HIP_P_GAIN = 220.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

_CHASSIS_NAME_PATTERN = re.compile(r"\w+_chassis_\w+")
_MOTOR_NAME_PATTERN = re.compile(r"\w+_hip_motor_\w+")
_KNEE_NAME_PATTERN = re.compile(r"\w+_lower_leg_\w+")
_TOE_NAME_PATTERN = re.compile(r"jtoe\d*")

URDF_FILENAME = "laikago/laikago_toes_limits.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203

UPPER_ANGLE_JOINT = [1.0471975512, 3.92699081699, -0.610865238198 ,
                     1.0471975512, 3.92699081699, -0.610865238198,
                     1.0471975512, 3.92699081699, -0.610865238198,
                     1.0471975512, 3.92699081699, -0.610865238198,]
LOWER_ANGLE_JOINT = [-0.872664625997, -0.523598775598, -2.77507351067 ,
                     -0.872664625997, -0.523598775598, -2.77507351067,
                     -0.872664625997, -0.523598775598, -2.77507351067,
                     -0.872664625997, -0.523598775598, -2.77507351067]

#UPPER_ANGLE_JOINT = np.array([0, 0.67,-1.25]*4) + 0.01
#LOWER_ANGLE_JOINT = np.array([0, 0.67,-1.25]*4) - 0.01

class Laikago(minitaur.Minitaur):
  """A simulation for the Laikago robot."""
  
  # ACTION_CONFIG = [
  #     locomotion_gym_config.ScalarField(name="motor_angle_0", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_1", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_2", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_3", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_4", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_5", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_6", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_7", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_8", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_9", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_10", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND),
  #     locomotion_gym_config.ScalarField(name="motor_angle_11", upper_bound=UPPER_BOUND, lower_bound=LOWER_BOUND)
  # ]
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="motor_angle_0", upper_bound=UPPER_ANGLE_JOINT[0], lower_bound=LOWER_ANGLE_JOINT[0]),
      locomotion_gym_config.ScalarField(name="motor_angle_1", upper_bound=UPPER_ANGLE_JOINT[1], lower_bound=LOWER_ANGLE_JOINT[1]),
      locomotion_gym_config.ScalarField(name="motor_angle_2", upper_bound=UPPER_ANGLE_JOINT[2], lower_bound=LOWER_ANGLE_JOINT[2]),
      locomotion_gym_config.ScalarField(name="motor_angle_3", upper_bound=UPPER_ANGLE_JOINT[3], lower_bound=LOWER_ANGLE_JOINT[3]),
      locomotion_gym_config.ScalarField(name="motor_angle_4", upper_bound=UPPER_ANGLE_JOINT[4], lower_bound=LOWER_ANGLE_JOINT[4]),
      locomotion_gym_config.ScalarField(name="motor_angle_5", upper_bound=UPPER_ANGLE_JOINT[5], lower_bound=LOWER_ANGLE_JOINT[5]),
      locomotion_gym_config.ScalarField(name="motor_angle_6", upper_bound=UPPER_ANGLE_JOINT[6], lower_bound=LOWER_ANGLE_JOINT[6]),
      locomotion_gym_config.ScalarField(name="motor_angle_7", upper_bound=UPPER_ANGLE_JOINT[7], lower_bound=LOWER_ANGLE_JOINT[7]),
      locomotion_gym_config.ScalarField(name="motor_angle_8", upper_bound=UPPER_ANGLE_JOINT[8], lower_bound=LOWER_ANGLE_JOINT[8]),
      locomotion_gym_config.ScalarField(name="motor_angle_9", upper_bound=UPPER_ANGLE_JOINT[9], lower_bound=LOWER_ANGLE_JOINT[9]),
      locomotion_gym_config.ScalarField(name="motor_angle_10", upper_bound=UPPER_ANGLE_JOINT[10], lower_bound=LOWER_ANGLE_JOINT[10]),
      locomotion_gym_config.ScalarField(name="motor_angle_11", upper_bound=UPPER_ANGLE_JOINT[11], lower_bound=LOWER_ANGLE_JOINT[11])
  ]

  def __init__(self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=True,
      time_step=0.001,
      action_repeat=33,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=True
  ):
    self._urdf_filename = urdf_filename

    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
                ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
    motor_kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
                ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]

    motor_torque_limits = None # jp hack
    
    super(Laikago, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_model_class=laikago_motor.LaikagoMotorModel,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        on_rack=on_rack,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter)

    return

  def _LoadRobotURDF(self):
    laikago_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          laikago_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          laikago_urdf_path, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()

    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)
    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)

  def GetHipPositionsInBaseFrame(self):
    return _DEFAULT_HIP_POSITIONS

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue

    return contacts

  def ComputeJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Because of the default rotation in the Laikago URDF, we need to reorder
    # the rows in the Jacobian matrix.
    return super(Laikago, self).ComputeJacobian(leg_id)[(2, 0, 1), :]

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "hip_motor_2_chassis_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
      elif "upper_leg_2_hip_motor_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
      elif "lower_leg_2_upper_leg_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)
      self._pybullet_client.resetJointState(
          self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._knee_link_ids = []
    self._foot_link_ids = []

    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if _CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_ids.append(joint_id)
      elif _MOTOR_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif _KNEE_NAME_PATTERN.match(joint_name):
        self._knee_link_ids.append(joint_id)
      elif _TOE_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._knee_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)
    self._foot_link_ids.extend(self._knee_link_ids)

    self._chassis_link_ids.sort()
    self._motor_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

    return

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    # The Laikago URDF assumes the initial pose of heading towards z axis,
    # and belly towards y axis. The following transformation is to transform
    # the Laikago initial orientation to our commonly used orientation: heading
    # towards -x direction, and z axis is the up direction.
    init_orientation = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])
    return init_orientation

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).N
      motor_control_mode: A MotorControlMode enum.
    """
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)

    super(Laikago, self).ApplyAction(motor_commands, motor_control_mode)
    return

  def _ClipMotorCommands(self, motor_commands):
    """Clips motor commands.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).

    Returns:
      Clipped motor commands.
    """

    # clamp the motor command by the joint limit, in case weired things happens
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

  @classmethod
  def GetConstants(cls):
    del cls
    return laikago_constants

