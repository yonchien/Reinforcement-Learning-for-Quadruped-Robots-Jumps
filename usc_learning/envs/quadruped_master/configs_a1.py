"""Defines the A1 robot related constants and URDF specs."""
#import collections
import numpy as np
import re
import pybullet as pyb
import os, inspect

from usc_learning.utils.robot_utils import FormatFRtoAll, format_FR_ranges_to_all
import usc_learning.envs as envs

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
currentdir = os.path.dirname(currentdir)
""" """
envs_base_path = os.path.dirname(inspect.getfile(envs))
URDF_ROOT = os.path.join(envs_base_path, 'pyb_data/')
#URDF_ROOT = os.path.join(currentdir, 'pyb_data/')

# *WARNING* the fixed hip joints are very bad for pybullet!!
#URDF_FILENAME = "a1_description/urdf/a1.urdf"

# removed fix links at hips, very weird behavior in pybullet
# URDF_FILENAME = "a1_description/urdf/a1_rm_fixhips.urdf"

# this file uses .stl mashes instead of .dae, which results in ~5x speed up
# URDF_FILENAME = "a1_description/urdf/a1_rm_fixhips_stl.urdf"
# fix hip position 0.08505 to 0.0838
URDF_FILENAME = "a1_description/urdf/a1_rm_fixhips_stl_v2.urdf"
# URDF_FILENAME = "a1_new/a1.urdf"

##################################################################################
# Default base and joint positions
##################################################################################

NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1]
# INIT_POSITION = [0, 0, 0.35]
#INIT_POSITION = [0, 0, 0.35]
# test to get base closer to desired height 0.3
INIT_POSITION = [0, 0, 0.305]
#INIT_POSITION = [0, 0, 0.2]
# INIT_POSITION = [0, 0, 0.4]

# height at which robot is considered fallen
IS_FALLEN_HEIGHT = 0.23 #0.15
IS_FALLEN_HEIGHT = 0.2
IS_FALLEN_HEIGHT = 0.15

# For consistency with Laikago
INIT_ORIENTATION = (0, 0, 0, 1) 
_, INIT_ORIENTATION_INV = pyb.invertTransform(
        position=[0, 0, 0], orientation=INIT_ORIENTATION)

# default angles (for init)
DEFAULT_HIP_ANGLE = 0
DEFAULT_THIGH_ANGLE = 0.65 # from aliengo.py had 0.65
DEFAULT_CALF_ANGLE = -1.2 # from aliengo.py had -1.2

DEFAULT_HIP_ANGLE = 0
# DEFAULT_THIGH_ANGLE = 0.85 # from aliengo.py had 0.65
# DEFAULT_CALF_ANGLE = -1.45 # from aliengo.py had -1.2
DEFAULT_THIGH_ANGLE = np.pi/4 # from aliengo.py had 0.65, 0.785
DEFAULT_CALF_ANGLE = -np.pi/2 # from aliengo.py had -1.2, -1.57

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

# joint offsets (for aliengo/a1 not used)
HIP_JOINT_OFFSET = 0.0
THIGH_JOINT_OFFSET = 0.0
CALF_JOINT_OFFSET = 0.0

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_OFFSETS = np.array([  HIP_JOINT_OFFSET, 
                            THIGH_JOINT_OFFSET,
                            CALF_JOINT_OFFSET] * NUM_LEGS)

# The hip joint location in the CoM frame.
HIP_POSITIONS = np.array([
    [ 0.183, -0.047, 0],
    [ 0.183,  0.047, 0],
    [-0.183, -0.047, 0],
    [-0.183,  0.047, 0],
])
# if include hip offset
hip_offset = 0.065
HIP_POSITIONS_RESET = np.array([
    [ 0.183, -0.047-hip_offset, 0],
    [ 0.183,  0.047+hip_offset, 0],
    [-0.183, -0.047-hip_offset, 0],
    [-0.183,  0.047+hip_offset, 0],
])

# Kinematics
HIP_LINK_LENGTH = 0.0838
THIGH_LINK_LENGTH = 0.2
CALF_LINK_LENGTH = 0.2

# actual hip location foot should be under should be at 0.0838+0.
# DEFAULT_LOCAL_EE_POS = np.concatenate(FormatFRtoAll(np.array([0.18357448, -0.12875846, -0.28373343])))
DEFAULT_LOCAL_EE_POS = FormatFRtoAll(np.array([0.18357448, -0.12875846, -0.28373343]))

##################################################################################
# Actuation limits/gains, position, and velocity limits
##################################################################################
# joint limits
UPPER_ANGLE_JOINT = np.array([ 0.802851455917,  4.18879020479, -0.916297857297 ] * NUM_LEGS)
LOWER_ANGLE_JOINT = np.array([-0.802851455917, -1.0471975512 , -2.69653369433  ] * NUM_LEGS)

# ranges we may actually care about
# UPPER_ANGLE_JOINT = np.array([ 0.802851455917,  DEFAULT_THIGH_ANGLE + np.pi/4, DEFAULT_CALF_ANGLE + 0.6 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.802851455917,  DEFAULT_THIGH_ANGLE - np.pi/4, DEFAULT_CALF_ANGLE - 0.6 ] * NUM_LEGS)
# this was used  previously with success
# UPPER_ANGLE_JOINT = np.array([ 0.4,  DEFAULT_THIGH_ANGLE + np.pi/4, DEFAULT_CALF_ANGLE + 0.6 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.4,  DEFAULT_THIGH_ANGLE - np.pi/4, DEFAULT_CALF_ANGLE - 0.6 ] * NUM_LEGS)
# pi/4 = 0.78
# UPPER_ANGLE_JOINT = np.array([ 0.4, 2, -1 ] * NUM_LEGS)
# # LOWER_ANGLE_JOINT = np.array([-0.4,-np.pi/4, -2.69653369433 ] * NUM_LEGS)
# UPPER_ANGLE_JOINT = np.array([ 0.4,  DEFAULT_THIGH_ANGLE + np.pi/4, -0.916297857297 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.4,  DEFAULT_THIGH_ANGLE - np.pi/4, -2.69653369433 ] * NUM_LEGS)
# UPPER_ANGLE_JOINT = np.array([ 0.4,  DEFAULT_THIGH_ANGLE + np.pi/4, -1 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.4,  DEFAULT_THIGH_ANGLE - np.pi/4, -2.6965 ] * NUM_LEGS)
# pretty good
# UPPER_ANGLE_JOINT = np.array([ 0.4,  1.75, -1 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.4,  -0.2, -2.6965 ] * NUM_LEGS)
# pretty good
# UPPER_ANGLE_JOINT = np.array([ 0.4,  DEFAULT_THIGH_ANGLE + 0.5, DEFAULT_CALF_ANGLE + 0.6 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.4,  DEFAULT_THIGH_ANGLE - 0.5, DEFAULT_CALF_ANGLE - 0.6 ] * NUM_LEGS)
"""
From best impedance:
=========================== Max joint angles
[0.0977 1.45 -1.08 
0.337 1.5 -1.12 
0.0084 1.92 -1.13 
0.457 1.68 -1.46]
=========================== Min joint angles
[-0.323 -0.032 -2.28  
-0.0772 -0.203 -2.34 
-0.388 0.601 -2.18 
-0.00551 0.0248 -2.25]


"""
# UPPER_ANGLE_JOINT = np.array([0.0977, 1.45, -1.08 , 0.337, 1.5, -1.12 ,0.0084, 1.92, -1.13 ,0.457 ,1.68, -1.46])
# LOWER_ANGLE_JOINT = np.array([-0.323 ,-0.032, -2.28  , -0.0772, -0.203, -2.34  , -0.388 ,0.601 ,-2.18  , -0.00551 ,0.0248, -2.25])

# UPPER_ANGLE_JOINT = np.array([ 0.4,  DEFAULT_THIGH_ANGLE + 0.6, DEFAULT_CALF_ANGLE + 0.6 ] * NUM_LEGS)
# LOWER_ANGLE_JOINT = np.array([-0.4,  DEFAULT_THIGH_ANGLE - 0.6, DEFAULT_CALF_ANGLE - 0.6 ] * NUM_LEGS)


# max / min ranges from gazebo
# UPPER_ANGLE_JOINT = np.array([0.1162 ,   1.3578 ,  -1.0630  ,  0.4858  ,  1.2317  , -1.1725  ,  0.0727  ,  1.7587  , -1.1495  ,  0.4856  ,  1.4253  , -1.3253])
# LOWER_ANGLE_JOINT = np.array([-0.4857 ,  -0.1856 ,  -2.6965 ,  -0.0545 ,  -0.2005  , -2.6965 ,  -0.4856 ,   0.7671 , -2.6965  , -0.1249  ,  0.0333 ,  -2.6965 ])
# from loads in pybullet
# UPPER_ANGLE_JOINT = np.array([ 0.4 ,   2 ,  -1 ]*4)
# LOWER_ANGLE_JOINT = np.array([-0.4 ,  -0.3 ,  -2.69 ]*4)

# quick PD test
#UPPER_ANGLE_JOINT = INIT_JOINT_ANGLES + 0.01
#LOWER_ANGLE_JOINT = INIT_JOINT_ANGLES - 0.01

# torque and velocity limits (Unitree - bug)
TORQUE_LIMITS   = np.asarray( [20.0, 55.0, 55.0] * 4 )
VELOCITY_LIMITS = np.asarray( [52.4, 28.6, 28.6] * 4 ) 

# torque and velocity limits (USC ROS Repo)
TORQUE_LIMITS   = np.asarray( [33.5] * 12 )
VELOCITY_LIMITS = np.asarray( [21.0] * 12 ) 

# test to see if it works for A1 (worked for Laikago)
# TORQUE_LIMITS   = np.asarray( [100] * 12 )
# VELOCITY_LIMITS = np.asarray( [100] * 12 ) 

# CHECK FOR A1?
# MOTOR_KP = [220.0, 220.0, 220.0] * NUM_LEGS
# MOTOR_KD = [0.3, 2.0, 2.0] * NUM_LEGS
# MOTOR_KP = [70.0, 180.0, 200.0] * NUM_LEGS
# MOTOR_KD = [3.0, 3.0, 3.0] * NUM_LEGS

# this is actually pretty good for PD stand - trained basically as fast as IMP
MOTOR_KP = [220.0, 220.0, 220.0] * NUM_LEGS
MOTOR_KD = [2.0, 2.0, 2.0] * NUM_LEGS
# MOTOR_KD = [0.2, 0.2, 0.2] * NUM_LEGS

MOTOR_KP = [500.0, 500.0, 500.0] * NUM_LEGS
MOTOR_KD = [3.0, 3.0, 3.0] * NUM_LEGS

# MOTOR_KP = [1000.0, 1000.0, 1000.0] * NUM_LEGS
# MOTOR_KD = [30.0, 30.0, 30.0] * NUM_LEGS

MOTOR_KP = [100.0, 100.0, 100.0] * NUM_LEGS
MOTOR_KD = [1.0, 2.0, 2.0] * NUM_LEGS
# MOTOR_KD = [1.0, 1.0, 2.0] * NUM_LEGS

# MOTOR_KP = [15] * NUM_LEGS*3
# MOTOR_KD = [0.2] * NUM_LEGS*3
# MOTOR_KP = [20] * NUM_LEGS*3
# MOTOR_KD = [0.5] * NUM_LEGS*3
# MOTOR_KP = [50] * NUM_LEGS*3
# MOTOR_KD = [1] * NUM_LEGS*3

# Regulates the joint angle change when in position control mode.
# NOTE: this should be a function of the time step - i.e. smaller time step means this 
#   should be smaller
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.1

##################################################################################
# Impedance control ranges: VERIFY reasonable, these are in LEG frame
##################################################################################
# IMPEDANCE - really in LEG FRAME so x,y limits should be around 0
FR_low_leg = [-0.1, -0.13 , -0.4]
FR_upp_leg = [ 0.1, -0.03, -0.25]

# new test
FR_low_leg = [-0.2, -HIP_LINK_LENGTH - 0.05, -0.35]
FR_upp_leg = [ 0.2, -HIP_LINK_LENGTH + 0.05, -0.15 ]

# for GAS
# FR_low_leg = [-0.05, -HIP_LINK_LENGTH - 0.05, -0.4]
# FR_upp_leg = [ 0.05, -HIP_LINK_LENGTH + 0.05, -0.2]

# new test , for standing
# FR_low_leg = [-0.1, -0.05, -0.4]
# FR_upp_leg = [ 0.1,  0.05, -0.25]

# increase ranges
# FR_low_leg = [-0.2, -0.2, -0.4 ]
# FR_upp_leg = [ 0.2,  0.0, -0.15]

# FR_low_leg = [-0.2, -0.2, -0.35]
# FR_upp_leg = [ 0.2,  0.0, -0.15]

# limits, for z are ~= -0.35
# FR_low_leg = [-0.3, -0.25, -0.35]
# FR_upp_leg = [ 0.3,  0.05, -0.15]

# new test
FR_low_leg = [-0.2, -HIP_LINK_LENGTH - 0.05, -0.33]
FR_upp_leg = [ 0.2, -HIP_LINK_LENGTH + 0.05, -0.15 ]

# new smaller action space to try to get over obstacles more easily 
# FR_low_leg = [-0.1, -HIP_LINK_LENGTH - 0.05, -0.33]
# FR_upp_leg = [ 0.1, -HIP_LINK_LENGTH + 0.05, -0.15 ]

# Get foot ranges in respective leg frames
IK_POS_LOW_LEG, IK_POS_UPP_LEG = format_FR_ranges_to_all(FR_low_leg,FR_upp_leg)

# check upper bounds on normal force with A1
MIN_NORMAL_FORCE = np.array([-60]*4)
MAX_NORMAL_FORCE = np.array([10]*4)

# MIN_NORMAL_FORCE = np.array([-200]*4) # have also tested -200, -100, -60 ...
# MAX_NORMAL_FORCE = np.array([10]*4)   # 10

# MIN_NORMAL_FORCE = np.array([-100]*4) # 100 didn't work...
# MAX_NORMAL_FORCE = np.array([10]*4)   # 10

"""
Notes:
------
MIN_NORMAL_FORCE 300 with kpCartesian [500,500,500] not good
MIN_NORMAL_FORCE 200 with kpCartesian [500,500,500] better 
"""

#---------------- IMPDEDANCE_3D_FORCE --------------------------
MIN_NORMAL_FORCE_3D = np.array([-50,-50,-200]*4)
MAX_NORMAL_FORCE_3D = np.array([ 50, 50, 10]*4)
# MIN_NORMAL_FORCE_3D = np.array([-100,-100,-200]*4)
# MAX_NORMAL_FORCE_3D = np.array([ 100, 100, 10]*4)
# MIN_NORMAL_FORCE_3D = np.array([-30,-30,-150]*4)
# MAX_NORMAL_FORCE_3D = np.array([ 30, 30, 10]*4)

#---------------- Choose XY for MPC ----------------------------
# XY positions only, for MPC
IK_POS_LOW_MPC = np.zeros(8)
IK_POS_UPP_MPC = np.zeros(8)
# for MPC chosen RL foot positions
MPC_FR_low_leg = [-0.1, - 0.05, -0.35]
MPC_FR_upp_leg = [ 0.1,   0.05, -0.25]
# Get foot ranges in respective leg frames
MPC_IK_POS_LOW_LEG, MPC_IK_POS_UPP_LEG = format_FR_ranges_to_all(MPC_FR_low_leg,MPC_FR_upp_leg)
for i in range(4):
    IK_POS_LOW_MPC[2*i:2*i+2] = MPC_IK_POS_LOW_LEG[3*i:3*i+2]
    IK_POS_UPP_MPC[2*i:2*i+2] = MPC_IK_POS_UPP_LEG[3*i:3*i+2]

# 
#---------------- IMPDEDANCE_FULL ----------------------------
# full impedance limits (pDes, vDes, F)
imp_vel = 1*np.ones(12)
FULL_IMPEDANCE_LOW = np.concatenate((IK_POS_LOW_LEG, -imp_vel, MIN_NORMAL_FORCE_3D))
FULL_IMPEDANCE_UPP = np.concatenate((IK_POS_UPP_LEG,  imp_vel, MAX_NORMAL_FORCE_3D))

# FULL_IMPEDANCE_UPP = np.array([ 1, 1, 0]*4 + [ 3]*12 + [ 300]*12)
# FULL_IMPEDANCE_LOW = np.array([-1,-1,-1]*4 + [-3]*12 + [-300]*12)
#-------------------------------------------------------------------------------
# Gains
kpCartesian = np.diag([500,500,500])
kdCartesian = np.diag([10,10,10])

# new test higher
# kpCartesian = np.diag([1000,1000,1000])
# kdCartesian = np.diag([30,30,30])
# kpCartesian = np.diag([500,500,800])
# kdCartesian = np.diag([10,10,20])
# kpCartesian = np.diag([800,800,800])
# kdCartesian = np.diag([15,15,15])
# kdCartesian = np.diag([20,20,20])

#kpCartesian = np.diag([500,500,500])
#kdCartesian = np.diag([20,20,20]) # 10 better than 30

# kpCartesian = np.diag([2000,2000,2000])
# kdCartesian = np.diag([30,30,30])

# kpCartesian = np.diag([2000,2000,2000])
# kdCartesian = np.diag([100,100,100])

# kpCartesian = np.diag([2000,2000,2000])
# kdCartesian = np.diag([50,50,50])

# kpCartesian = np.diag([600,600,600])
# kdCartesian = np.diag([10,10,10])

# for paper
kpCartesian = np.diag([700,700,700])
kdCartesian = np.diag([12,12,12])

# # TEST HIGHER GAINS
# kpCartesian = np.diag([1500,1500,1500])
# # kpCartesian = np.diag([2000,2000,2000])
# kdCartesian = np.diag([25,25,25])

######################################################################################
########################### POSITIONS
NOMINAL_JOINT_POS = INIT_JOINT_ANGLES
# how much to increase these limits? they are not in center of ranges

########################### TORQUES
NOMINAL_TORQUES = np.zeros(12)

########################### IMPEDANCE
NOMINAL_IMPEDANCE_POS = np.concatenate(FormatFRtoAll(np.array([0., -HIP_LINK_LENGTH, -0.3])))
#NOMINAL_IMPEDANCE_POS = np.array([0., 0., -0.3]*4)
NOMINAL_IMPEDANCE_FN = np.array([-30.5]*4) # verify..

# robot_config.NOMINAL_IMPEDANCE_POS = NOMINAL_JOINT_POS
# robot_config.NOMINAL_TORQUES = NOMINAL_TORQUES
# robot_config.NOMINAL_IMPEDANCE_POS = NOMINAL_IMPEDANCE_POS
# robot_config.NOMINAL_IMPEDANCE_FN = NOMINAL_IMPEDANCE_FN

##################################################################################
# INVERSE KINEMATICS ranges: in BODY frame 
##################################################################################
FR_low = [0.1, -0.15 , -0.4]
FR_upp = [0.3, -0.05, -0.25]

IK_POS_LOW, IK_POS_UPP = format_FR_ranges_to_all(FR_low,FR_upp)

#lower limits for null space
LL = list(LOWER_ANGLE_JOINT)
#upper limits for null space
UL = list(UPPER_ANGLE_JOINT)
#joint ranges for null space
JR = list(np.array(UPPER_ANGLE_JOINT) - np.array(LOWER_ANGLE_JOINT))
#restposes for null space
RP =[0.03, 0.65, -1.2, 
    -0.03, 0.65, -1.2, 
     0.03, 0.65, -1.2, 
    -0.03, 0.65, -1.2]



##################################################################################
# Imitation env jumping gains 
##################################################################################
kps = np.asarray([400,500,500]*4)
kds = np.asarray([7,5,5] * 4)

# tests (not bad)
# kps = np.asarray([100,100,100]*4)
# kds = np.asarray([3,3,3] * 4)
# ok
# kps = np.asarray([100,50,50]*4)
# kds = np.asarray([3,1.25,1.25] * 4)

# kps = np.asarray([100,50,50]*4)
# kds = np.asarray([3,1.5,1.5] * 4)

# good
kps = np.asarray([400,400,400]*4)
kds = np.asarray([7,5,5] * 4) # 4 good here as well

kps = np.asarray([300,300,300]*4)
kds = np.asarray([3,3,3] * 4)

# worse performance ?
# kps = np.asarray([100,100,100]*4)
# kds = np.asarray([2,2,2] * 4)
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
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_j\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_j\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")

# new urdf
# _CHASSIS_NAME_PATTERN = re.compile(r"\w*imu_joint\w*")
# _HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
# _THIGH_NAME_PATTERN = re.compile(r"\w+_upper_j\w+")
# _CALF_NAME_PATTERN = re.compile(r"\w+_lower_j\w+")
# _FOOT_NAME_PATTERN = re.compile(r"\w+toe\w+")