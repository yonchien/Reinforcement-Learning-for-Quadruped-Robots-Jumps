"""Defines the aliengo robot related constants and URDF specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import re
import pybullet as pyb
import os, inspect

from usc_learning.utils.robot_utils import FormatFRtoAll

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
currentdir = os.path.dirname(currentdir)
""" """
URDF_ROOT = os.path.join(currentdir, 'pyb_data/')

HIP_LIMITS_FLAG = True # whether to limit the hip range (in urdf)

if HIP_LIMITS_FLAG:
    URDF_FILENAME = "aliengo/urdf/aliengo_hipLimits.urdf" # updated URDF values AND limiting hips in [-pi,pi]
    URDF_FILENAME = "aliengo/urdf/aliengo_hipLimits_pi2.urdf"
else:
    URDF_FILENAME = "aliengo/urdf/aliengo.urdf"

##################################################################################
# Default base and joint positions
##################################################################################

NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.45]

# height at which robot is considered fallen
IS_FALLEN_HEIGHT = 0.25 

# For consistency with Laikago
INIT_ORIENTATION = (0, 0, 0, 1) 
_, INIT_ORIENTATION_INV = pyb.invertTransform(
        position=[0, 0, 0], orientation=INIT_ORIENTATION)

# default angles (for init)
DEFAULT_HIP_ANGLE = 0
DEFAULT_THIGH_ANGLE = 0.65 # from aliengo.py had 0.65
DEFAULT_CALF_ANGLE = -1.2 # from aliengo.py had -1.2

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
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 
                             1, 1, 1, 1, 1, 1])

# joint offsets (for aliengo not used)
HIP_JOINT_OFFSET = 0.0
THIGH_JOINT_OFFSET = 0.0
CALF_JOINT_OFFSET = 0.0

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_OFFSETS = np.array([  HIP_JOINT_OFFSET, 
                            THIGH_JOINT_OFFSET,
                            CALF_JOINT_OFFSET] * NUM_LEGS)

# The hip joint location in the CoM frame.
HIP_POSITIONS = np.array([
    [ 0.2399, -0.051, 0],
    [ 0.2399,  0.051, 0],
    [-0.2399, -0.051, 0],
    [-0.2399,  0.051, 0],
])
# if include hip offset
hip_offset = 0.083
HIP_POSITIONS_RESET = np.array([
    [ 0.2399, -0.051-hip_offset, 0],
    [ 0.2399,  0.051+hip_offset, 0],
    [-0.2399, -0.051-hip_offset, 0],
    [-0.2399,  0.051+hip_offset, 0],
])

##################################################################################
# Actuation limits/gains, position, and velocity limits
##################################################################################
if HIP_LIMITS_FLAG:
    UPPER_ANGLE_JOINT = np.array([ 1.2217304764,  3.14, -0.645771823238] * NUM_LEGS)
    LOWER_ANGLE_JOINT = np.array([-1.2217304764, -3.14, -2.77507351067 ] * NUM_LEGS)
    """ For pi2 urdf"""
    UPPER_ANGLE_JOINT = np.array([ 1.2217304764,  1.57, -0.645771823238] * NUM_LEGS)
    LOWER_ANGLE_JOINT = np.array([-1.2217304764, -1.57, -2.77507351067 ] * NUM_LEGS) 
else:

    UPPER_ANGLE_JOINT = np.array([ 1.0471975512,    3.92699081699,  -0.610865238198 ] * NUM_LEGS)
    LOWER_ANGLE_JOINT = np.array([-0.872664625997, -0.523598775598, -2.77507351067  ] * NUM_LEGS)

    UPPER_ANGLE_JOINT =  np.pi*np.ones(12)
    LOWER_ANGLE_JOINT = -np.pi*np.ones(12)

    UPPER_ANGLE_JOINT =  np.ones(12)
    LOWER_ANGLE_JOINT = -np.ones(12)


TORQUE_LIMITS = np.asarray( [44,44,55] * 4 ) # from aliengo
VELOCITY_LIMITS = np.asarray( [20,20,16] * 4 ) # from aliengo

# these worked
MOTOR_KP = [220.0, 220.0, 220.0] * NUM_LEGS
MOTOR_KD = [0.3, 2.0, 2.0] * NUM_LEGS

# MOTOR_KP = [70.0, 180.0, 200.0] * NUM_LEGS
# MOTOR_KD = [3.0, 3.0, 3.0] * NUM_LEGS

# MOTOR_KP = [70.0, 500.0, 500.0] * NUM_LEGS
# MOTOR_KD = [2.0, 20.0, 20.0] * NUM_LEGS

# Regulates the joint angle change when in position control mode.
# NOTE: this should be a function of the time step - i.e. smaller time step means this 
#   should be smaller
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.05


##################################################################################
# Inverse kinematics and force control ranges: VERIFY reasonable
##################################################################################
# scale possible area?
# for limb 0 (FR)
# x: (0.1, 0.5)
# y: (-0.2, -0.03)
# z: (-0.1, -0.5) ?
#FR_low = [0.1, -0.2 , -0.5]
#FR_upp = [0.5, -0.03, -0.1]

# IMPEDANCE - really in LEG FRAME so x limit should be around 0
FR_low = [-0.1, -0.13 , -0.45]
FR_upp = [ 0.1, -0.03, -0.35]

FL_low = [ FR_low[0], -FR_upp[1], FR_low[2]]
FL_upp = [ FR_upp[0], -FR_low[1], FR_upp[2]]

RR_low = [-FR_upp[0],  FR_low[1], FR_low[2]]
RR_upp = [-FR_low[0],  FR_upp[1], FR_upp[2]]

RL_low = [-FR_upp[0], -FR_upp[1], FR_low[2]]
RL_upp = [-FR_low[0], -FR_low[1], FR_upp[2]]

IK_POS_LOW = np.concatenate(([FR_low,FL_low,RR_low,RL_low]))
IK_POS_UPP = np.concatenate(([FR_upp,FL_upp,RR_upp,RL_upp]))
# this is wrong, gets the max/min wrong
# IK_POS_LOW = np.concatenate(FormatFRtoAll(FR_low))
# IK_POS_UPP = np.concatenate(FormatFRtoAll(FR_upp))

MIN_NORMAL_FORCE = np.array([-70]*4)
MAX_NORMAL_FORCE = np.array([10]*4)

##################################################################################
# Hip, thigh, calf strings, naming conventions, etc.
##################################################################################

# Can be different from the motors, although for laikago they are the same list.
JOINT_NAMES = (
    # front right leg
    "FR_hip_joint", 
    "FR_thigh_joint", 
    "FR_calf_joint",
    # front left leg
    "FL_hip_joint", 
    "FL_thigh_joint", 
    "FL_calf_joint",
    # rear right leg
    "RR_hip_joint", 
    "RR_thigh_joint", 
    "RR_calf_joint",
    # rear left leg
    "RL_hip_joint", 
    "RL_thigh_joint", 
    "RL_calf_joint",
)
MOTOR_NAMES = JOINT_NAMES

# _CHASSIS_NAME_PATTERN = re.compile(r"\w+_chassis_\w+")
# _MOTOR_NAME_PATTERN = re.compile(r"\w+_hip_motor_\w+")
# _KNEE_NAME_PATTERN = re.compile(r"\w+_lower_leg_\w+")
# _TOE_NAME_PATTERN = re.compile(r"jtoe\d*")

# standard across all robots
_CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")


##################################################################################
# Below should not be needed (leftover from Laikago)
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



