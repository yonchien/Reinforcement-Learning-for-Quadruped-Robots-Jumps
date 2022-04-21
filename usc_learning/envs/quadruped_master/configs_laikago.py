"""Defines the laikago robot related constants and URDF specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, inspect
import collections
import math
import numpy as np
import re
import pybullet as pyb
import pybullet_data
from usc_learning.utils.robot_utils import FormatFRtoAll, format_FR_ranges_to_all
import usc_learning.envs as envs

URDF_ROOT = pybullet_data.getDataPath()
URDF_FILENAME = "laikago/laikago_toes_limits.urdf"

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# #parentdir = os.path.dirname(os.path.dirname(currentdir))
# currentdir = os.path.dirname(currentdir)
# """ """
# envs_base_path = os.path.dirname(inspect.getfile(envs))
# URDF_ROOT = os.path.join(envs_base_path, 'pyb_data/')
# URDF_FILENAME = "laikago/urdf/laikago.urdf"
##################################################################################
# Default base and joint positions
##################################################################################

NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]

# height at which robot is considered fallen
IS_FALLEN_HEIGHT = 0.3 

# For laikago_toes_limits.urdf, 
# The Laikago URDF assumes the initial pose of heading towards z axis,
# and belly towards y axis. The following transformation is to transform
# the Laikago initial orientation to our commonly used orientation: heading
# towards -x direction, and z axis is the up direction.
# Will be default to (0, 0, 0, 1) once the new laikago_toes_zup.urdf checked in.
INIT_ORIENTATION = pyb.getQuaternionFromEuler([math.pi / 2.0, 0, math.pi / 2.0])
# INIT_ORIENTATION = (0, 0, 0, 1) 
_, INIT_ORIENTATION_INV = pyb.invertTransform(
        position=[0, 0, 0], orientation=INIT_ORIENTATION)

# default angles (for init)
DEFAULT_HIP_ANGLE = 0
DEFAULT_THIGH_ANGLE = 0.67
DEFAULT_CALF_ANGLE = -1.25

""" WITHOUT silly offsets """
# DEFAULT_HIP_ANGLE = 0
# DEFAULT_THIGH_ANGLE = 0.07
# DEFAULT_CALF_ANGLE = -0.59

# Note this matches the Laikago SDK/control convention, but is different from
# URDF's internal joint angles which needs to be computed using the joint
# offsets and directions. The conversion formula is (sdk_joint_angle + offset) *
# joint direction.
INIT_JOINT_ANGLES = np.array([  DEFAULT_HIP_ANGLE, 
                                DEFAULT_THIGH_ANGLE, 
                                DEFAULT_CALF_ANGLE] * NUM_LEGS)
INIT_MOTOR_ANGLES = INIT_JOINT_ANGLES
#INIT_MOTOR_ANGLES = INIT_MOTOR_ANGLES * JOINT_DIRECTIONS

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_DIRECTIONS = np.array([-1, 1, 1, 1, 1, 1, 
                             -1, 1, 1, 1, 1, 1])

# joint offsets (needed? or can just shift action distribution)
HIP_JOINT_OFFSET = 0.0
THIGH_JOINT_OFFSET = -0.6
CALF_JOINT_OFFSET = 0.66

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_OFFSETS = np.array([  HIP_JOINT_OFFSET, 
                            THIGH_JOINT_OFFSET,
                            CALF_JOINT_OFFSET] * NUM_LEGS)

# The hip joint location in the CoM frame.
HIP_POSITIONS = (
    ( 0.21, -0.1157, 0),
    ( 0.21,  0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21,  0.1157, 0),
)

# these are NOT ACCURATE
HIP_LINK_LENGTH = 0.0817145 #0 0.242889
# THIGH_LINK_LENGTH = 0.25
# CALF_LINK_LENGTH = 0.25 

# HIP_LINK_LENGTH = 0.08 #0 0.242889
# HIP_LINK_LENGTH = 0.037
# HIP_LINK_LENGTH = 0
THIGH_LINK_LENGTH = 0.25 #0.25
CALF_LINK_LENGTH = 0.25 
##################################################################################
# Actuation limits/gains, position, and velocity limits
##################################################################################
UPPER_ANGLE_JOINT = np.array([ 1.0471975512,    3.92699081699,  -0.610865238198 ] * NUM_LEGS)
LOWER_ANGLE_JOINT = np.array([-0.872664625997, -0.523598775598, -2.77507351067  ] * NUM_LEGS)

# UPPER_ANGLE_JOINT =  INIT_MOTOR_ANGLES + .01
# LOWER_ANGLE_JOINT = INIT_MOTOR_ANGLES - .01

UPPER_ANGLE_JOINT =  2*np.pi*np.ones(12)
LOWER_ANGLE_JOINT = -2*np.pi*np.ones(12)

# UPPER_ANGLE_JOINT =  np.ones(12)
# LOWER_ANGLE_JOINT = -np.ones(12)

jnt_fctr = .5
UPPER_ANGLE_JOINT = INIT_MOTOR_ANGLES + jnt_fctr*np.ones(12)
LOWER_ANGLE_JOINT = INIT_MOTOR_ANGLES - jnt_fctr*np.ones(12)


# TORQUE_LIMITS = np.asarray( [44,44,55] * 4 ) # from aliengo
# VELOCITY_LIMITS = np.asarray( [20,20,16] * 4 ) # from aliengo

TORQUE_LIMITS = np.asarray( [100] * 12 ) # from urdf
VELOCITY_LIMITS = np.asarray( [100] * 12 ) # from urdf

# from unitree github
# TORQUE_LIMITS = np.asarray( [20,55,55] * 4 ) # from aliengo
# VELOCITY_LIMITS = np.asarray( [52.4,28.6,28.6] * 4 ) # from aliengo

MOTOR_KP = [220.0, 220.0, 220.0] * NUM_LEGS
MOTOR_KD = [0.3, 2.0, 2.0] * NUM_LEGS

# MOTOR_KP = [1000]*12
# MOTOR_KD = [50] * 12

# Regulates the joint angle change when in position control mode.
# NOTE: this should be a function of the time step - i.e. smaller time step means this 
#   should be smaller
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.1




##################################################################################
# Impedance control ranges: VERIFY reasonable, these are in LEG frame
##################################################################################
# IMPEDANCE - really in LEG FRAME so x,y limits should be around 0

# new test
# FR_low_leg = [-0.2, -HIP_LINK_LENGTH - 0.05, -0.5]
# FR_upp_leg = [ 0.2, -HIP_LINK_LENGTH + 0.05, -0.3]

# to make sure limits are correct
FR_low_leg = [-0.01,  -HIP_LINK_LENGTH - 0.001, -0.3]
FR_upp_leg = [ 0.01,  -HIP_LINK_LENGTH + 0.001, -0.3]

FR_low_leg = [-0.2,  -HIP_LINK_LENGTH - 0.05, -0.45]
FR_upp_leg = [ 0.2,  -HIP_LINK_LENGTH + 0.05, -0.3]

# Get foot ranges in respective leg frames
IK_POS_LOW_LEG, IK_POS_UPP_LEG = format_FR_ranges_to_all(FR_low_leg,FR_upp_leg)

kpCartesian = np.diag([800,800,800])
kdCartesian = np.diag([20,20,20])

# kpCartesian = np.diag([2000,2000,2000])
# kdCartesian = np.diag([50,50,50])
# just to try
NOMINAL_IMPEDANCE_POS = np.concatenate(FormatFRtoAll(np.array([0., -HIP_LINK_LENGTH, -0.4])))
NOMINAL_IMPEDANCE_FN = np.array([-50]*4) # verify..

##################################################################################
# Hip, thigh, calf strings, naming conventions, etc.
##################################################################################

# Can be different from the motors, although for laikago they are the same list.
JOINT_NAMES = (
    # front right leg
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    # front left leg
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    # rear right leg
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    # rear left leg
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
)
MOTOR_NAMES = JOINT_NAMES

# _CHASSIS_NAME_PATTERN = re.compile(r"\w+_chassis_\w+")
# _MOTOR_NAME_PATTERN = re.compile(r"\w+_hip_motor_\w+")
# _KNEE_NAME_PATTERN = re.compile(r"\w+_lower_leg_\w+")
# _TOE_NAME_PATTERN = re.compile(r"jtoe\d*")

# standard across all robots
# _HIP_NAME_PATTERN = re.compile(r"\w+_chassis_\w+")
# _THIGH_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
# _CALF_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
# _FOOT_NAME_PATTERN = re.compile(r"jtoe\d*")

_CHASSIS_NAME_PATTERN = re.compile(r"chassis")
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_motor_2_chassis_\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_upper_leg_2_hip_motor_\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_lower_leg_2_upper_leg_\w+")
_FOOT_NAME_PATTERN = re.compile(r"jtoe\w+")

""" For new URDF """
# _CHASSIS_NAME_PATTERN = re.compile(r"trunk")
# _HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
# _THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_j\w+")
# _CALF_NAME_PATTERN = re.compile(r"\w+_calf_j\w+")
# _FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")
##################################################################################
# Below should not be needed
##################################################################################
LEG_NAMES = (
    "front_right",
    "front_left",
    "rear_right",
    "rear_left",
)

LEG_ORDER = (
    "front_right",
    "front_left",
    "back_right",
    "back_left",
)

END_EFFECTOR_NAMES = (
    "jtoeFR",
    "jtoeFL",
    "jtoeRR",
    "jtoeRL",
)

MOTOR_NAMES = JOINT_NAMES
MOTOR_GROUP = collections.OrderedDict((
    (LEG_NAMES[0], JOINT_NAMES[0:3]),
    (LEG_NAMES[1], JOINT_NAMES[3:6]),
    (LEG_NAMES[2], JOINT_NAMES[6:9]),
    (LEG_NAMES[3], JOINT_NAMES[9:12]),
))



