"""This file implements the imitation gym environment of a quadruped. 
See:
traj_motion_data.py for reading in the trajectory and its interface
sim_opt_traj_v2.py is an example of just running the trajectory
"""

import os, inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0, parentdir)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import time, datetime
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
#import pybullet
#import pybullet_utils.bullet_client as bc
import pybullet_data
#import os

#from usc_learning.envs.my_minitaur import my_minitaur_env_randomizer
#import usc_learning.envs.quadruped_master.quadruped as quadruped
from pkg_resources import parse_version
import re

# Should only be using A1 for jumping

import usc_learning.envs.quadruped_master.configs_a1 as robot_config
from usc_learning.envs.quadruped_master import quadruped_gym_env #import QuadrupedGymEnv
import usc_learning.envs.quadruped_master.quadruped as quadruped
from usc_learning.imitation_tasks.traj_motion_data import TrajTaskMotionData
import usc_learning
from usc_learning.envs.quadruped_master.quadruped_gym_env import VIDEO_LOG_DIRECTORY

opt_data = "data/jumpingFull_A1.csv"
opt_data = 'data/jumpingFull_A1_1ms_h50_d60.csv'

# with cartesian 
opt_data = 'data2/jumpingFull_A1_1ms_h50_d60.csv'
# backflip
opt_data = 'data3/backflipFull_A1_1ms_h0_d-50.csv'

opt_trajs_path = 'data3'
opt_trajs_path = 'data4'
opt_trajs_path = 'data5'





print('*'*80)
print('opt trajs path', opt_trajs_path)

# zeros first to account for resets
# TEST_TIME_HEIGHTS_F_R = np.array([ [0.005, 0.005, 0.05 , 0.005, 0.025, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.01] ,
#                                    [0.005, 0.005, 0.005, 0.05 , 0.025, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.07]]).T
# TEST_TIME_HEIGHTS_F_R = np.array([ [0.005, 0.005, 0.05 , 0.005, 0.025, 0.01, 0.05, 0.06, 0.08, 0.09, 0.01, 0.01, 0.01, 0.01] ,
#                                    [0.005, 0.005, 0.005, 0.05 , 0.025, 0.05, 0.01, 0.01, 0.01, 0.01, 0.09, 0.08, 0.07, 0.06]]).T

# .1m is too much
# TEST_TIME_HEIGHTS_F_R = np.array([ [0.005, 0.005, 0.05 , 0.005, 0.025, 0.01, 0.05, 0.06,  0.01] ,
#                                    [0.005, 0.005, 0.005, 0.05 , 0.025, 0.05, 0.01, 0.01,  0.06]]).T
TEST_TIME_HEIGHTS_F_R = np.array([ [0.005, 0.005, 0.05 , 0.005, 0.05,0.04, 0.04,0.04, 0.03] ,
                                   [0.005, 0.005, 0.005, 0.05 , 0.01,0.01, 0.01,0.01,0.01]]).T

TEST_TIME_HEIGHTS_F_R = np.array([ [0.01, 0.01] ,
                                   [0.01, 0.01]]).T
TEST_TIME_HEIGHTS_F_R = np.array([ [0.01, 0.05, 0.01, 0.1] ,
                                   [0.05, 0.01, 0.1 , 0.01]]).T
TEST_IDX = 0
TEST_TIME_HEIGHTS_F_R = []

# def get_opt_trajs(path):
#   """ Returns all optimal trajectories in directory. """
#   try:
#     files =  os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), path)
    
#   except:
#     files = os.listdir(path)
#   paths = [os.path.join(path, basename) for basename in files if (basename.endswith('.csv') and 'cartesian' not in basename)]
#   return paths #return max(paths, key=os.path.getctime)

# print('*'*80)
# print('opt trajs path', opt_trajs_path)
# print('get ', get_opt_trajs(opt_trajs_path))

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01

FLIGHT_TIME_START = 0.8


CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD", "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE" ]

TASK_MODES = ["DEFAULT", 
              "DEFAULT_CARTESIAN",
              "IL_ADD_CARTESIAN_P",
              "IL_ADD_CARTESIAN_P_PREJUMP_ONLY",
              "IL_ADD_CARTESIAN_P_JOINT",
              "IL_ADD_CARTESIAN_P_JOINT_PREJUMP_ONLY",
              "IL_ADD_TORQUE" , 
              "IL_ADD_Qdes", 
              "IL_ADD_Qdes_dQdes", 
              "IL_ADD_Qdes_PREJUMP_ONLY"]

TASK_ENVS = ["FULL_TRAJ", # output full trajectory deltas
            ]
"""
Given the optimal trajectory tracking controller: 

  tau_ff = kp*(q_des-q) + kd*(dq_des-dq) + torque_des

Cartesian PD

  tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel)) + tau_ff

Choice of learning to adjust each of these? 
"""

# Modes (Many: TODO)
# DEFAULT: simulate opt trajectory (no learning)
# IL_ADD_TORQUE: policy adds torque 
#   -offset ffwd torque
#   i.e. Action is: tau + tau_RL; (let RL choose an offset torque to the optimal data, for example in range [-20,20])
# IL_LEARN_TORQUE_DES: replace ffwd torque with RL
#   -tau_new = kp*(q_des-q) + kd(dq_des-dq) + tau_RL; (let RL choose the feedforward torque)
# IL_LEARN_PD_DES: replace PD controller with RL
#   -tau_new = torque_des + tau_RL; (let RL act as a PD controller)
# IL_LEARN_GAINS: learn Kp and Kd (RL tunes gains only)
# IL_ADD_Qdes: policy adds delta_qdes 
#   -offset desired position to optimization qdes
# IL_ADD_Qdes_dQdes: policy adds (delta_qdes, delta_dqdes)
#   -offset desired (positions, velocities) to optimization (qdes, dqdes)
# IL - PD: learn from scratch to imitate traj

# MODE [TODO find name] PREJUMP_ONLY
# - let RL adjust the trajetory while on the ground, then FFWD jumping joints, see where it ends up, reward based on that 

# if INIT in name, then only before X (s) and/or on ground

# OBS space 
# MOTION_IMITATION_EXTENDED: current robot state compared to opt traj, and lookahead
# MOTION_IMITATION_EXTENDED_CONTACTS:  also include current foot contact booleans
# MOTION_IMITATION_EXTENDED2_CONTACTS: also include foot pos/vel

class ImitationGymEnv(quadruped_gym_env.QuadrupedGymEnv):
  """ The imitation gym environment for a quadruped. """
  NUM_SUBSTEPS = 10


  def __init__(self,
      robot_config=robot_config,
      traj_filename=opt_data,
      use_multiple_trajs=True, # if use multiple trajectories 
      opt_trajs_path=opt_trajs_path,
      useTrajCartesianData=True, # whether to use Cartesian PD control - allows better tracking
      sparse_rewards=False,
      isRLGymInterface=True, # True means running RL and should scale actions, should be false if running tests
      enable_clip_motor_commands=True, 
      time_step=0.001, # 0.01
      action_repeat=10,
      sensors=None,
      set_kp_gains="LOW", # either LOW / HIGH
      motor_control_mode="TORQUE",
      task_env="FULL_TRAJ", # RL "action" is entire trajectory as 1 pass through NN
      task_env_num=8, # how many time sections to learn offsets for pre_jump (so 0.8 / x)
      task_mode="IL_ADD_CARTESIAN_P_JOINT_PREJUMP_ONLY",
      observation_space_mode="MOTION_IMITATION_EXTENDED2_CONTACTS",
      #observation_space_mode="MOTION_IMITATION_EXTENDED_HISTORY", # CHUONG
      land_and_settle=False,
      traj_num_lookahead_steps=4,
      accurate_motor_model_enabled=True,
      rl_action_high=1.0,       # action high for RL
      grow_action_space=False,  # use growing action spaces
      scale_gas_action=True,    # True: scale action range (i.e. smaller than [-1,1]), False: scale high/low limits being scaled to
      control_latency=0.002,
      hard_reset=True, # Should be left as True, even if may be slightly slower to reload env
      on_rack=False,
      enable_action_interpolation=True, # does nothing
      enable_action_filter=False ,  # this is good to have True (debatable)
      render=False,
      record_video=False,
      env_randomizer=None,
      randomize_dynamics=True,
      add_terrain_noise=False,
      test_index=TEST_IDX,
      test_heights=TEST_TIME_HEIGHTS_F_R,
      **kwargs # any extras from legacy
      ):
    """
      Args:
      robot_config: The robot config file, should be A1 for jumping.
      traj_filename: Which traj to try to track.
      isRLGymInterface: If the gym environment is being run as RL or not. Affects
        if the actions should be scaled.
      enable_clip_motor_commands: TODO\
      time_ste: Sim time step. WILL BE CHANGED BY NUM_SUBSTEPS.
      action_repeat: The number of simulation steps before actions are applied.
      sensors: TODO.
      motor_control_mode: Whether to use torque control, PD, control, etc.
      task_mode: See descriptions above. Execute traj, add to traj, or pure RL.
      traj_num_lookahead_steps: how many future trajectory points to include 
        (see get_observation for details on which)
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
        Should always be true.
      control_latency: TODO.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the quadruped back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the quadruped on rack. This is only used to debug
        the walking gait. In this mode, the quadruped's base is hanged midair so
        that its walking gait is clearer to visualize.
      enable_action_interpolation: TODO
      enable_action_filter: TODO
      render: Whether to render the simulation.
      env_randomizer: TO ADD. An EnvRandomizer to randomize the physical properties
        during reset().
    """
    self._sensory_hist_len = 2
    def get_opt_trajs(path):
      """ Returns all optimal trajectories in directory. """
      # try:
      #   files =  os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), path)
      # except:
      path = os.path.join(os.path.dirname(os.path.abspath(usc_learning.imitation_tasks.__file__)),path)
      files = os.listdir(path)
      paths = [os.path.join(path, basename) for basename in files if (basename.endswith('.csv') and 'cartesian' not in basename)]
      return paths 

    self._use_multiple_trajs = use_multiple_trajs
    if use_multiple_trajs:
      self.opt_traj_filenames = get_opt_trajs(opt_trajs_path)
      self.num_opt_trajs = len(self.opt_traj_filenames)
      self.curr_opt_traj = 0
      self._traj_task = TrajTaskMotionData(filename=self.opt_traj_filenames[self.curr_opt_traj % self.num_opt_trajs],dt=0.001,useCartesianData=useTrajCartesianData)
      # iterate through each trajectory to get max/min observations
      traj_task_max_range = np.amax(self._traj_task.full_state, axis=1)
      traj_task_min_range = np.amin(self._traj_task.full_state, axis=1)
      for i in range(self.num_opt_trajs):
        traj_task = TrajTaskMotionData(filename=self.opt_traj_filenames[i],dt=0.001,useCartesianData=useTrajCartesianData)
        traj_task_max_range = np.maximum( traj_task_max_range, np.amax(traj_task.full_state, axis=1) )
        traj_task_min_range = np.minimum( traj_task_min_range, np.amin(traj_task.full_state, axis=1) )
      self.traj_task_min_range = traj_task_min_range
      self.traj_task_max_range = traj_task_max_range
    else:
      # desired trajectory to track (need to initialize first)
      self._traj_task = TrajTaskMotionData(filename=traj_filename,dt=0.001,useCartesianData=useTrajCartesianData)
      self.traj_task_max_range = np.amax(self._traj_task.full_state, axis=1)
      self.traj_task_min_range = np.amin(self._traj_task.full_state, axis=1)
    self._useTrajCartesianData = useTrajCartesianData
    self._task_mode = task_mode
    self._traj_num_lookahead_steps = traj_num_lookahead_steps
    self._enable_action_filter = enable_action_filter
    self._traj_task_index = 0
    self._motor_control_mode = motor_control_mode
    #self._observation_space_mode = observation_space_mode
    self._sparse_rewards = sparse_rewards
    self._task_env = task_env
    self._task_env_num = task_env_num
    self._set_kp_gains = set_kp_gains
    self._land_and_settle = land_and_settle
    self.TEST_TIME_HEIGHTS_F_R = test_heights 
    self.TEST_IDX = test_index
    if land_and_settle:
      self.opt_traj_filenames = get_opt_trajs('test_data2')#get_opt_trajs('test_data') # test_data2 test_paper_vid
      self.num_opt_trajs = len(self.opt_traj_filenames)
      self.curr_opt_traj = 0


    self.save_real_base_xyz = []
    self.save_traj_base_xyz = []
    self.save_real_base_rpy = []
    self.save_traj_base_rpy = []

    if self._task_env == "FULL_TRAJ":
      action_dim = 12 * task_env_num #( FLIGHT_TIME_START / time_step * action_repeat )
      action_repeat = int(FLIGHT_TIME_START / (time_step * task_env_num ))
    elif self._task_mode in ["IL_ADD_TORQUE", "IL_ADD_Qdes", \
                          "IL_ADD_Qdes_PREJUMP_ONLY","IL_ADD_CARTESIAN_P","IL_ADD_CARTESIAN_P_JOINT", \
                          "IL_ADD_CARTESIAN_P_JOINT_PREJUMP_ONLY", "IL_ADD_CARTESIAN_P_PREJUMP_ONLY"]:
      action_dim = 12
    elif self._task_mode in ["IL_ADD_Qdes_dQdes"]:
      action_dim = 24
    elif self._motor_control_mode in ["PD","TORQUE"]:
      action_dim = 12
    elif self._motor_control_mode in ["IMPEDANCE"]:
      action_dim = 16
    else:
      raise ValueError("motor control mode" + motor_control_mode + "not implemented yet.")
    self._action_dim = action_dim
    #self.setupObsAndActionSpaces()

    super(ImitationGymEnv, self).__init__(
      urdf_root=pybullet_data.getDataPath(), #pyb_data.getDataPath() # careful_here 
      robot_config=robot_config,
      isRLGymInterface=isRLGymInterface, 
      time_step=time_step,
      action_repeat=action_repeat,
      task_env=task_env,
      # observation_noise_stdev=0.0,
      # motor_velocity_limit=np.inf,
      # leg_model_enabled=True,
      accurate_motor_model_enabled=accurate_motor_model_enabled,
      motor_control_mode=motor_control_mode,
      observation_space_mode=observation_space_mode,
      rl_action_high=rl_action_high,       
      grow_action_space=grow_action_space,  
      scale_gas_action=scale_gas_action,  
      hard_reset=hard_reset,
      render=render,
      record_video=record_video,
      env_randomizer=env_randomizer,
      randomize_dynamics=randomize_dynamics,
      add_terrain_noise=add_terrain_noise) 
    
  def setupObsAndActionSpaces(self):
    """Set up observation and action spaces for imitation learning. """
    # observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
    # observation_low = (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)

    if self._observation_space_mode == "MOTION_IMITATION_EXTENDED_CONTACTS":
      obs_high =(self._robot.GetExtendedObservationUpperBound() + OBSERVATION_EPS)
      obs_low = (self._robot.GetExtendedObservationLowerBound() - OBSERVATION_EPS)
    elif self._observation_space_mode == "MOTION_IMITATION_EXTENDED2_CONTACTS":
      obs_high =(self._robot.GetExtendedObservation2UpperBound() + OBSERVATION_EPS)
      obs_low = (self._robot.GetExtendedObservation2LowerBound() - OBSERVATION_EPS)
    # elif self._observation_space_mode == "MOTION_IMITATION_EXTENDED2_CONTACTS_HISTORY":
    #   obs_high =(self._robot.GetExtendedObservation2UpperBound_History() + OBSERVATION_EPS) # size is 73*hist_len
    #   obs_low = (self._robot.GetExtendedObservation2LowerBound_History() - OBSERVATION_EPS)
    #   print("Chuong obs_high size:", obs_high.shape)
    else:
      raise ValueError ('observation space not defined')

    if self._task_env == "FULL_TRAJ":
      obs_high = np.concatenate( (obs_high, self.traj_task_max_range))
      obs_low  = np.concatenate( (obs_low,  self.traj_task_min_range))

    else:
      for i in range(self._traj_num_lookahead_steps):
        # obs_high = np.concatenate( (obs_high, 
        #                 np.amax(self._traj_task.full_state, axis=1) + OBSERVATION_EPS ))
        # obs_low  = np.concatenate( (obs_low, 
        #                 np.amin(self._traj_task.full_state, axis=1) - OBSERVATION_EPS))
        obs_high = np.concatenate( (obs_high, self.traj_task_max_range))
        obs_low  = np.concatenate( (obs_low,  self.traj_task_min_range))


    #if self._observation_space_mode == "MOTION_IMITATION_EXTENDED_CONTACTS":
    if "CONTACTS" in self._observation_space_mode:
      obs_high = np.concatenate((obs_high, np.ones(4) + OBSERVATION_EPS))
      obs_low = np.concatenate((obs_low, np.zeros(4) - OBSERVATION_EPS))

    self._action_bound = 1
    self._last_action_rl = np.zeros(self._action_dim)
    #action_high = np.array([self._action_bound] * action_dim)
    if self._GAS and self._scale_GAS_action: # likely in lower range than [-1,1]
      action_high = np.array([self._rl_action_bound] * self._action_dim)
    else:
      action_high = np.array([self._action_bound] * self._action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
    
  def setupObsSpace(self):
    """Set up observation and action spaces for imitation learning. """
    # observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
    # observation_low = (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)

    if self._observation_space_mode == "MOTION_IMITATION_EXTENDED_CONTACTS":
      obs_high =(self._robot.GetExtendedObservationUpperBound() + OBSERVATION_EPS)
      obs_low = (self._robot.GetExtendedObservationLowerBound() - OBSERVATION_EPS)
    elif self._observation_space_mode == "MOTION_IMITATION_EXTENDED2_CONTACTS":
      obs_high =(self._robot.GetExtendedObservation2UpperBound() + OBSERVATION_EPS) # size is 73
      obs_low = (self._robot.GetExtendedObservation2LowerBound() - OBSERVATION_EPS)
      print("obs_high size:", obs_high.shape)
    elif self._observation_space_mode == "MOTION_IMITATION_EXTENDED2_CONTACTS_HISTORY":
      obs_high =(self._robot.GetExtendedObservation2UpperBound_History() + OBSERVATION_EPS) # size is 73*hist_len
      obs_low = (self._robot.GetExtendedObservation2LowerBound_History() - OBSERVATION_EPS)
      # print("Chuong obs_high size:", obs_high.shape)
    else:
      raise ValueError ('observation space not defined')

    if self._task_env == "FULL_TRAJ":
      # obs_high = np.concatenate((obs_high, self.traj_task_max_range)) # size is 73*hist_len+72
      # obs_low  = np.concatenate((obs_low,  self.traj_task_min_range))
      obs_high = np.concatenate((obs_high, np.tile(self.traj_task_max_range, self._sensory_hist_len))) # Chuong: size is 73*hist_len+72*his_len?
      obs_low  = np.concatenate((obs_low,  np.tile(self.traj_task_min_range, self._sensory_hist_len)))
      # print("obs:", obs_high.shape)

    else:
      for i in range(self._traj_num_lookahead_steps):
        # obs_high = np.concatenate( (obs_high, 
        #                 np.amax(self._traj_task.full_state, axis=1) + OBSERVATION_EPS ))
        # obs_low  = np.concatenate( (obs_low, 
        #                 np.amin(self._traj_task.full_state, axis=1) - OBSERVATION_EPS))
        obs_high = np.concatenate( (obs_high, self.traj_task_max_range))
        obs_low  = np.concatenate( (obs_low,  self.traj_task_min_range))


    #if self._observation_space_mode == "MOTION_IMITATION_EXTENDED_CONTACTS":
    if "CONTACTS" in self._observation_space_mode:
      # obs_high = np.concatenate((obs_high, np.ones(4) + OBSERVATION_EPS))
      # obs_low = np.concatenate((obs_low, np.zeros(4) - OBSERVATION_EPS))
      obs_high = np.concatenate((obs_high, np.array([1, 1, 1, 1]*self._sensory_hist_len) + OBSERVATION_EPS))
      obs_low = np.concatenate((obs_low,   np.array([1, 1, 1, 1]*self._sensory_hist_len) - OBSERVATION_EPS))

    self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

  def setupActionSpace(self):
    self._action_bound = 1
    self._last_action_rl = np.zeros(self._action_dim)
    #action_high = np.array([self._action_bound] * action_dim)
    if self._GAS and self._scale_GAS_action: # likely in lower range than [-1,1]
      action_high = np.array([self._rl_action_bound] * self._action_dim)
    else:
      action_high = np.array([self._action_bound] * self._action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
  ######################################################################################
  # Reset
  ######################################################################################
  def _load_plane(self):
    """ Load plane at specific point, should be specific width/height. """
    #print('LOADED STAIRS')
    plane_pos = [.9, 0, .5]
    plane_scaling = 0.005#0.0025
    plane_pos = [100.4, 0, .5]
    plane_scaling = 1 #0.0025
    plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root,
                                          basePosition=plane_pos,
                                          baseOrientation=(0,0,0,1),
                                          globalScaling=plane_scaling)
    self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 0, 0, 1])

    plane_pos = [.4,0,-99.5]
    plane_orn = [0,-np.pi/2,0]
    plane_orn = self._pybullet_client.getQuaternionFromEuler(plane_orn)
    plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root,
                                          basePosition=plane_pos,
                                          baseOrientation=plane_orn,
                                          globalScaling=plane_scaling)
    self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 0, 0, 1])


    # add vertical plane
    # plane_pos = [.4,0,0]
    # plane_orn = [0,-np.pi/2,0]
    # plane_orn = self._pybullet_client.getQuaternionFromEuler(plane_orn)
    # plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root,
    #                                       basePosition=plane_pos,
    #                                       baseOrientation=plane_orn,
    #                                       globalScaling=plane_scaling)
    # self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 0, 0, 1])

    # just replace with pybullet version?

    # colBoxId = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_PLANE)
    # col=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=colBoxId, 
    #                                       basePosition=[1,1,2])
    #                               #halfExtents=[0.5, 0.5, 0.25])
    # print(col)
    # self._pybullet_client.changeVisualShape(col, -1, rgbaColor=[0, 1, 0, 1])
    #sys.exit()
    # below some notes on adding platforms/stairs in pybullet

    # box_x = 1
    # box_y = 1
    # box_z = 0.3
    # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
    #           halfExtents=[box_x/2,box_y/2,box_z/2])
    # sth=0.25
    # block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [box_x-0.1,0,box_z/2],baseOrientation=[0,0,0,1])

    # boxHalfLength = 2.5
    # boxHalfWidth = 2.5
    # boxHalfHeight = 0.2
    # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
    #           halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    #mass = 1
    #block=p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                        basePosition = [-2,0,-0.1],baseOrientation=[0.0,0.1,0.0,1])
    # sth=0.25
    # block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [2.75,0,-0.2+1*sth],baseOrientation=[0.0,0.0,0.0,1])
    # block3=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [2.75+0.33,0,-0.2+2*sth],baseOrientation=[0.0,0.0,0.0,1])
    # block4=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [2.75+0.66,0,-0.2+3*sth],baseOrientation=[0.0,0.0,0.0,1])
    # block5=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [2.75+0.99,0,-0.2+4*sth],baseOrientation=[0.0,0.0,0.0,1])
    # box_x = 1
    # box_y = 1
    # box_z = 0.5
    # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
    #           halfExtents=[box_x/2,box_y/2,0.01])
    # sth=0.25
    # block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [box_x-0.2,0,box_z],baseOrientation=[0,0,0,1])

  def reset(self):
    #print(self._time_step,self._action_repeat)
    if self._hard_reset:
      #print('hard reset')
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      self.plane = plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
      #self._load_plane()
      # self.add_box()
      #self.add_box2()

      if self._use_multiple_trajs:
        # load next traj
        # self.curr_opt_traj += 1
        # if self.curr_opt_traj >= self.num_opt_trajs:
        #   self.curr_opt_traj = 0
        self.curr_opt_traj = np.random.randint(0,self.num_opt_trajs)
        self._traj_task = TrajTaskMotionData(filename=self.opt_traj_filenames[self.curr_opt_traj],dt=0.001,useCartesianData=self._useTrajCartesianData)
        # add box according to height, distance
        #_CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
        curr_traj_name = os.path.basename(self.opt_traj_filenames[self.curr_opt_traj])
        jump_height = float(curr_traj_name[20:22])/100
        jump_dist = float(curr_traj_name[24:26])/100

        # means we are in 3 digit, 100
        if jump_dist < .3:
          jump_dist = float(curr_traj_name[24:27])/100
        if self._is_render and jump_dist > 0.5:
          print('Current traj is:', curr_traj_name)
          print('Adding box at h', jump_height, 'd', jump_dist)
          self.add_box_at(jump_height,jump_dist)
      
      self._robot_config.INIT_POSITION = [0, 0, 0.2]
      rand_front = 0
      rand_rear = 0
      if self._randomize_dynamics:
        self._robot_config.INIT_POSITION = [0, 0, 0.2]
        USE_4_BOXES = False
        if USE_4_BOXES:
          #for i in range(4):
          z_heights = 0.025*np.random.random(4)
          self.add_box_each_foot(z_heights)
          min_mu = 1
          ground_mu_k_each= min_mu+(1-min_mu)*np.random.random(4)
          self._pybullet_client.changeDynamics(self.box_FR, -1, lateralFriction=ground_mu_k_each[0])
          self._pybullet_client.changeDynamics(self.box_FL, -1, lateralFriction=ground_mu_k_each[1])
          self._pybullet_client.changeDynamics(self.box_RR, -1, lateralFriction=ground_mu_k_each[2])
          self._pybullet_client.changeDynamics(self.box_RL, -1, lateralFriction=ground_mu_k_each[3])
        else:
          mult = 0.05
          mult = 0.1
          rand_front = mult*np.random.random()
          rand_rear =  mult*np.random.random()
          if self._land_and_settle:
            self.TEST_IDX = self.TEST_IDX % len(self.TEST_TIME_HEIGHTS_F_R)
            rand_front = self.TEST_TIME_HEIGHTS_F_R[self.TEST_IDX,0]
            rand_rear = self.TEST_TIME_HEIGHTS_F_R[self.TEST_IDX,1]
            self.TEST_IDX += 1
            print('======== USING HEIGHTS', rand_front, rand_rear, "TEST IDX", self.TEST_IDX)
          self.add_box_ff(z_height=rand_front)
          self.add_box_rr(z_height=rand_rear)
          min_mu = 0.5 #1
          min_mu = 0.5
          ground_mu_k_front = min_mu+(1-min_mu)*np.random.random()
          ground_mu_k_rear = min_mu+(1-min_mu)*np.random.random()
          self._pybullet_client.changeDynamics(self.box_ff, -1, lateralFriction=ground_mu_k_front)
          self._pybullet_client.changeDynamics(self.box_rr, -1, lateralFriction=ground_mu_k_rear)

      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._pybullet_client.setGravity(0, 0, -10)
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection
      self._robot = (quadruped.Quadruped(pybullet_client=self._pybullet_client,
                                         robot_config=self._robot_config,
                                         time_step=self._time_step,
                                         self_collision_enabled=self._self_collision_enabled,
                                         motor_velocity_limit=self._motor_velocity_limit,
                                         pd_control_enabled=self._pd_control_enabled,
                                         accurate_motor_model_enabled=acc_motor,
                                         motor_control_mode=self._motor_control_mode,
                                         motor_overheat_protection=motor_protect,
                                         on_rack=self._on_rack,
                                         render=self._is_render))

      
    else:
      #print('soft reset')
      self._robot.Reset(reload_urdf=False)
      #self._load_plane()

    if self._env_randomizer is not None:
      raise ValueError('Not implemented randomizer yet.')
      self._env_randomizer.randomize_env(self)

    base_pos = self._robot.GetBasePosition()
    #print('base pos before anything happens', self._robot.GetBasePosition())
    base_pos = list(base_pos)
    #if self._randomize_dynamics:
    base_pos[2] += max(rand_front,rand_rear) +.01
    base_orn = self._robot.GetBaseOrientation()
    self._robot.ResetBasePositionAndOrientation(base_pos,base_orn)
    if self._is_render:
      time.sleep(1)

    #print('base pos after reset to max + init', self._robot.GetBasePosition())
    # set initial condition from traj
    des_state = self._traj_task.get_state_at_index(0)
    des_torques = self._traj_task.get_torques_at_index(0)
    q_des = self._traj_task.get_joint_pos_from_state(des_state)
    self._robot.SetJointStates(np.concatenate((q_des,[0]*12)))

    ground_mu_k = 1#0.5#+0.5*np.random.random()
    self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
    if self._randomize_dynamics:
      # self._robot.RandomizePhysicalParams(mass_percent_change=0.05, inertia_percent_change=0.05,base_only=True)
      self._robot.RandomizePhysicalParams(mass_percent_change=0.2, base_only=False)
      # set random lateral friction for plane (also rotate?)
      # set friction/orientation for landing plane as well
      ground_mu_k = 1#0.5+0.5*np.random.random()
      self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
      if self._is_render:
        print('ground friction coefficient', ground_mu_k)

      
      # # reset plane orientation (not working?)
      # orn_noise = 0.05
      # #plane_orn = orn_noise/2*np.array([np.random.random(),np.random.random(),0]) - orn_noise
      # plane_orn = orn_noise * ( np.array([np.random.random(),np.random.random(),0]) - 0.5)
      # plane_orn = self._pybullet_client.getQuaternionFromEuler(plane_orn)
      # self._pybullet_client.resetBasePositionAndOrientation(self.plane, 
      #                                               [0,0,-2*orn_noise],
      #                                               plane_orn)

    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._traj_task_index = 0
    self._objectives = []
    # self.save_real_base_xyz = []
    # self.save_traj_base_xyz = []
    self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                     self._cam_pitch, [0, 0, 0])
    #if not self._torque_control_enabled: # self._motor_control_mode is not "TORQUE"
    # this is absolutely ridiculous, change to PD control mode to set initial position, then
    # set back..
    tmp_save_motor_control_mode_ENV = self._motor_control_mode
    tmp_save_motor_control_mode_ROB = self._robot._motor_control_mode
    tmp_save_motor_control_mode_MOT = self._robot._motor_model._motor_control_mode
    self._motor_control_mode = "PD"
    self._robot._motor_control_mode = "PD"
    self._robot._motor_model._motor_control_mode = "PD"
    if self._motor_control_mode is not "TORQUE":
      for _ in range(200):
        if self._pd_control_enabled or self._accurate_motor_model_enabled:
          #self._robot.ApplyAction([math.pi / 2] * 8)
          # self._robot.ApplyAction(self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS)
          self._robot.ApplyAction(q_des)
          #print('Motor angles', self._robot.GetMotorAngles())
          #print('Motor torques', self._robot.GetMotorTorques())
        self._pybullet_client.stepSimulation()
        if self._is_render:
          time.sleep(0.01)
    # set control mode back
    self._motor_control_mode = tmp_save_motor_control_mode_ENV
    self._robot._motor_control_mode = tmp_save_motor_control_mode_ROB
    self._robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT

    #print('base pos after init', self._robot.GetBasePosition())

    if self._is_record_video:
      curr_traj_name = os.path.basename(self.opt_traj_filenames[self.curr_opt_traj])
      self.recordVideoHelper(extra_filename=curr_traj_name[:-4])

    #noisy_obs = super(ImitationGymEnv, self).reset()
    # move robot to traj start and settle
    base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel = self.get_traj_state_at_time(0)
    init_base = np.concatenate((base_pos,base_orn))
    init_joints = np.concatenate((joint_pos,joint_vel))
    #self._robot.SetPoseAndSettle(init_base,init_joints)
    # now going to settle to sim start for real
    for _ in range(1000):
      des_state = self._traj_task.get_state_at_index(0)
      des_torques = self._traj_task.get_torques_at_index(0)
      q_des = self._traj_task.get_joint_pos_from_state(des_state)
      dq_des = self._traj_task.get_joint_vel_from_state(des_state)
      q = self._robot.GetMotorAngles()
      dq = self._robot.GetMotorVelocities()
      #u = self._opt_traj_pd_controller(q_des, q, dq_des, dq, des_torques)
      u = self._opt_traj_pd_controller(q_des, q, dq_des, dq, np.zeros(12))
      u = np.clip(u,
          -self._robot_config.TORQUE_LIMITS,
          self._robot_config.TORQUE_LIMITS)
      self._robot.ApplyAction(u)
      self._pybullet_client.stepSimulation()
      #time.sleep(0.1)
      #print(self._robot.GetBasePosition())

    #print('base pos after settle ', self._robot.GetBasePosition())

    self._last_action = self._robot._applied_motor_torque
    self._last_action_rl = np.zeros(self._action_dim)
    self._last_base_euler = self._robot.GetBaseOrientationRollPitchYaw()

    if np.any(np.abs(self._last_base_euler) > .5): 
      # means we probably fell over
      print('Bad start, reset. Pos is', self._robot.GetBasePosition(), 'ORN', self._last_base_euler)
      self.reset()

    return self._noisy_observation()


  ######################################################################################
  # Step 
  ######################################################################################
  def _action_filter(self,action,mode='rl'):
    """ Smooth actions. """
    lam = 0.1
    #lam = 0.1
    #return lam*action + (1-lam)*self._last_action
    return lam*action + (1-lam)*self._last_action_rl

  def _opt_traj_pd_controller(self, q_des, q, dq_des, dq, tau):
    """ Quick test controller to track opt traj, from sim_opt_traj_v2.py """
    #kps = np.asarray([300,500,500]*4)
    #kds = np.asarray([2,3,3] * 4)
    # THESE ARE THE GOOD ONES
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

    if self._set_kp_gains == "HIGH":
      # this may be better
      kps = self._robot_config.kps
      kds = self._robot_config.kds
    elif self._set_kp_gains == "LOW":
      kps = np.asarray([100,100,100]*4)
      kds = np.asarray([2,2,2] * 4)
    else:
      raise ValueError('check kp gains')


    # DO NOT USE, these are too low
    # kps = np.asarray([50,50,50]*4) # these are too low!
    # kds = np.asarray([1,1,1] * 4) 

    # kps = np.asarray([400,400,400]*4)
    # kds = np.asarray([4,4,4] * 4)

    return kps*(q_des-q) + kds*(dq_des-dq) - tau

  def _opt_traj_cartesian_pd_controller(self, foot_pos, foot_vel, foot_force):
    """ Compute Cartesian PD controller gains. 

    FOR NOW: 
    -ignore foot_force from opt
    -use default kpCartesian and kdCartesian as in normal impedance control, can update these as go on
    """
    actions = np.zeros(12)
    for j in range(4):
      actions[j*3:j*3+3] = self._robot._ComputeLegImpedanceControlActions(foot_pos[j*3:j*3+3],
                                          foot_vel[j*3:j*3+3],
                                          # np.diag([800,800,800]), #self._robot_config.kpCartesian
                                          # np.diag([30,30,30]), # self._robot_config.kdCartesian
                                          np.diag([500,500,500]), #self._robot_config.kpCartesian
                                          np.diag([10,10,10]), # self._robot_config.kdCartesian
                                          np.zeros(3),
                                          j)
    return actions

  def _get_default_imitation_action(self):
    # should this be get_sim_time + some dt ahead to line up correctly?
    # TODO: verify time
    curr_time = self.get_sim_time() + self._time_step #1e-4*np.random.random() #self._time_step#+ self._action_repeat*self._time_step
    #curr_time = self._env_step_counter * self._action_repeat * self._time_step
    #curr_time = self._sim_step_counter * self._time_step + self._time_step
    #print('get_sim_time', self.get_sim_time(), 'other', curr_time)
    #print(self._traj_task.get_duration())
    #print(self._action_repeat,self._time_step)

    USE_IDX = False
    if USE_IDX:
      # testing the index comparison
      self._traj_task_index += 1
      print('time sim', self.get_sim_time(), 'time traj index', self._traj_task.get_time_at_index(self._traj_task_index))

      # desired trajectory state at this time
      traj_base_pos_t, traj_base_orn_t, _,_, traj_joint_pos_t, traj_joint_vel_t = self.get_traj_state_at_time(curr_time)
      # feedforward torques
      des_torques_t = self._traj_task.get_torques_at_time(curr_time)

      # desired trajectory state at this time
      traj_base_pos, traj_base_orn, traj_base_linvel, \
      traj_base_angvel, traj_joint_pos, traj_joint_vel = self.get_traj_state_at_index(self._traj_task_index)
      # feedforward torques
      des_torques = self._traj_task.get_torques_at_index(self._traj_task_index)

      print( 'differences', np.linalg.norm(traj_base_pos_t-traj_base_pos),
                            np.linalg.norm(traj_base_orn_t-traj_base_orn),
                            np.linalg.norm(traj_joint_pos_t-traj_joint_pos),
                            np.linalg.norm(traj_joint_vel_t-traj_joint_vel) )
    else:

      # desired trajectory state at this time
      traj_base_pos, traj_base_orn, traj_base_linvel, \
      traj_base_angvel, traj_joint_pos, traj_joint_vel = self.get_traj_state_at_time(curr_time)
      # feedforward torques
      des_torques = self._traj_task.get_torques_at_time(curr_time)

    if self._is_render:
      # current robot state
      curr_base_pos, curr_base_orn, curr_base_linvel, \
      curr_base_angvel, curr_joint_pos, curr_joint_vel = self.get_robot_state()

      self.save_traj_base_xyz.extend(traj_base_pos)
      self.save_real_base_xyz.extend(curr_base_pos)
      self.save_traj_base_rpy.extend(traj_base_orn)
      self.save_real_base_rpy.extend(curr_base_orn)

      # print('traj ')
      # print(np.array(traj_base_pos),traj_base_orn)
      # print('curr')
      # print(np.array(curr_base_pos),curr_base_orn)
      # print('orientation')
      # print('RPY ', np.array(self._robot.GetBaseOrientationRollPitchYaw()))
      # print('quat', np.array(self._robot.GetBaseOrientation()))
      # print('comp', np.array(self._pybullet_client.getEulerFromQuaternion(self._robot.GetBaseOrientation())))
      # print('-'*30)


    #des_state = traj_task.get_state_at_index(i)
    #des_torques = self._traj_task.get_torques_at_index(i)
    q_des = traj_joint_pos
    dq_des = traj_joint_vel
    q = self._robot.GetMotorAngles()
    dq = self._robot.GetMotorVelocities()
    tau = des_torques
    return q_des, q, dq_des, dq, tau
    #return self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)

  def _get_cartesian_imitation_action(self):
    """ Get default foot pos/vel/force 
    Returns: foot_pos, foot_vel, foot_force (each 12 values)
    """
    curr_time = self.get_sim_time() + self._time_step
    return self.get_cartesian_traj_state_at_time(curr_time)

  def _transform_action_to_motor_command(self, action):
    """ Get action depeding on mode.
    DEFAULT: just set sequentially with PD controller on des traj state
    IL: 
    """
    if self._task_mode in TASK_MODES:
      # get optimal action from imitation data
      # note the opt action may be messed up from the additive torques we are supplying
      q_des, q, dq_des, dq, tau = self._get_default_imitation_action()

      if self._task_mode == "DEFAULT":
        # quick test
        # q = self._robot.GetMotorAngles()
        # motor_commands_min = (q - self._time_step * self._robot_config.VELOCITY_LIMITS)
        # motor_commands_max = (q + self._time_step * self._robot_config.VELOCITY_LIMITS)
        # motor_commands = np.clip(q_des, motor_commands_min, motor_commands_max)
        #action = self._opt_traj_pd_controller(motor_commands, q, dq_des, dq, tau)
        action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)

      elif self._task_mode == "DEFAULT_CARTESIAN":
        # As in Cheetah 3 jumping paper
        # joint PD w torque
        tau_ff = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
        # Cartesian PD
        foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
        tau = self._opt_traj_cartesian_pd_controller(foot_pos, foot_vel, foot_force)
        action = tau + tau_ff

      elif self._task_mode in ["IL_ADD_CARTESIAN_P", "IL_ADD_CARTESIAN_P_PREJUMP_ONLY"]:
        # As in Cheetah 3 jumping paper
        # joint PD w torque
        tau_ff = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
        # Cartesian PD
        foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
        # adding offset in leg frame
        delta_pos =  0.05 * action

        tau = self._opt_traj_cartesian_pd_controller(foot_pos+delta_pos, foot_vel, foot_force)
        action = tau + tau_ff

      elif self._task_mode in ["IL_ADD_CARTESIAN_P_JOINT", "IL_ADD_CARTESIAN_P_JOINT_PREJUMP_ONLY"]:
        # As in Cheetah 3 jumping paper, but now adding contribution in joint space as well
        # Cartesian PD
        foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
        # adding offset in leg frame
        # delta_pos =  0.05 * action
        delta_pos =  0.1 * action

        # get contribution in q_des as well 
        delta_q = np.zeros(12)
        for j in range(4):
          J, _ = self._robot._ComputeJacobianAndPositionUSC(j,specific_delta_q=q_des[j*3:j*3+3])
          delta_q[j*3:j*3+3] = J.T @ delta_pos[j*3:j*3+3]

        # joint PD w torque
        tau_ff = self._opt_traj_pd_controller(q_des+delta_q, q, dq_des, dq, tau)


        tau = self._opt_traj_cartesian_pd_controller(foot_pos+delta_pos, foot_vel, foot_force)
        action = tau + tau_ff

      elif self._task_mode in ["IL_ADD_Qdes", "IL_ADD_Qdes_PREJUMP_ONLY"]:
        # add position offsets only
        delta_q_des = 0.1 * action[:12] #self._robot.ScaleActionToPosRange(action[:12])
        # action = self._opt_traj_pd_controller(q_des+delta_q_des, q, dq_des+delta_dq_des, dq, tau)

        # clip desired motor
        # q = self._robot.GetMotorAngles()
        # motor_commands_min = (q - self._time_step * self._robot_config.VELOCITY_LIMITS)
        # motor_commands_max = (q + self._time_step * self._robot_config.VELOCITY_LIMITS)
        # motor_commands = np.clip(q_des+delta_q_des, motor_commands_min, motor_commands_max)
        # action = self._opt_traj_pd_controller(motor_commands, q, dq_des, dq, tau)
        #
        
        action = self._opt_traj_pd_controller(q_des+delta_q_des, q, dq_des, dq, tau)
      elif self._task_mode == "IL_ADD_Qdes_dQdes":
        # action is offsets to add for q and dq, should probably not be scaled to full value
        """
        [-1,1] is too high, lower these ranges here 
        maybe: 0.2, 0.1 ?
        check what optimization velocities are
        NEXT: try 0.1 for both, or just position
        -should gains be adjusted as well?
        """
        delta_q_des = 0.2 * action[:12] #self._robot.ScaleActionToPosRange(action[:12])
        delta_dq_des = 0.2 * action[12:] #self._robot.ScaleActionToVelRange(action[12:])
        action = self._opt_traj_pd_controller(q_des+delta_q_des, q, dq_des+delta_dq_des, dq, tau)
        #action = self._opt_traj_pd_controller(q_des+delta_q_des, q, dq_des, dq, tau)
      elif self._task_mode == "IL_ADD_TORQUE":
        # Add torque to whatever is coming from the optimal controller
        opt_action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
        # clip optimal action before adding, otherwise the addition might not affect
        opt_action = np.clip(opt_action,
                            -self._robot_config.TORQUE_LIMITS,
                            self._robot_config.TORQUE_LIMITS)
        # probably don't need to change that much (HOW MUCH?) - 
        #   -or, should this be the feedforward term? PD+RL
        #   -or instead of PD, just ffwd + RL
        add_action = self._robot._scale_helper(action,-20,20) 
        # clip to set back into limits (actually, will be clipped by motor)
        action = opt_action+add_action
      else:
        raise NotImplementedError('Task mode not implemented.')
    else: 
      # in pure RL mode trying to track the trajectory
      if self._motor_control_mode == "PD":
        action = self._robot.ScaleActionToPosRange(action)
      elif self._motor_control_mode == "IMPEDANCE":
        action = self._robot.ScaleActionToImpedanceControl(action)
      elif self._motor_control_mode == "TORQUE":
        action = self._robot.ScaleActionToTorqueRange(action)
      elif self._motor_control_mode is not "TORQUE":
        raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
    return action


  def _transform_action_GAS(self,action):
    """GAS - where we are changing the action space gradually to [-1,1] """

    if self._task_mode not in TASK_MODES:
      print('invalid task mode - doing pure RL')
      return self._transform_action_to_motor_command(action)

    # get optimal action from imitation data
    # note the opt action may be messed up from the additive torques we are supplying
    q_des, q, dq_des, dq, tau = self._get_default_imitation_action()

    if self._scale_GAS_action:
      # RL chooses action in [-1,1], which we then scale to limits with rl_action_bound
      action_mult = 0.2 * self._rl_action_bound
    else:
      # RL chooses action in [-rl_action_bound,+rl_action_bound] 
      action_mult = 0.2 * np.ones(len(action))

    #print('action ', action)
    if self._task_mode == "DEFAULT":
      # for debugging only
      action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
    elif self._task_mode == "IL_ADD_Qdes":
      # add position offsets only
      #delta_q_des = 0.2 * action[:12] #self._robot.ScaleActionToPosRange(action[:12])
      #action = self._opt_traj_pd_controller(q_des+delta_q_des, q, dq_des, dq, tau)
      action = self._opt_traj_pd_controller(q_des+action*action_mult, q, dq_des, dq, tau)
    elif self._task_mode == "IL_ADD_Qdes_dQdes":
      # add position and velocity offsets
      delta_q_des  = action[:12] * action_mult[:12] #self._robot.ScaleActionToPosRange(action[:12])
      delta_dq_des = action[12:] * action_mult[12:] #self._robot.ScaleActionToVelRange(action[12:])
      action = self._opt_traj_pd_controller(q_des+delta_q_des, q, dq_des+delta_dq_des, dq, tau)
    else:
      raise NotImplementedError('Task mode not implemented.')
    return action

  
  def step_full_traj(self,action):
    """ Step for full traj in action space. """

    # reshape action into number of offsets to use
    #reshaped_action = np.reshape(action,(12,-1))
    reshaped_action = np.reshape(action, (12, self._task_env_num))
    #print('reshaped_action', reshaped_action)

    for i in range(self._task_env_num):

      for _ in range(self._action_repeat): # check line 280
        proc_action = self._transform_action_to_motor_command(reshaped_action[:,i])

        self._robot.ApplyAction(proc_action)
        self._pybullet_client.stepSimulation()
        self._sim_step_counter += 1
        self._last_action = self._robot._applied_motor_torque
      
        if self._is_render:
          time.sleep(0.001)
          self._render_step_helper()

    # only learning on ground, so this MUST execute
    #_,_,_,feetInContactBool = self._robot.GetContactInfo()
    if True: #"PREJUMP_ONLY" in self._task_mode: # and self.get_sim_time() > FLIGHT_TIME_START : # and sum(feetInContactBool) == 0
      done = False
      total_reward = 0
      while not done: #self.get_sim_time() > FLIGHT_TIME_START and :
        #self._transform_action_to_motor_command(np.zeros(self._action_dim))
        q_des, q, dq_des, dq, tau = self._get_default_imitation_action()
        proc_action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)

        if "CARTESIAN" in self._task_mode:
          # if cartesian in name 
          # joint PD w torque
          #tau_ff = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
          # Cartesian PD
          foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
          tau = self._opt_traj_cartesian_pd_controller(foot_pos, foot_vel, foot_force)
          proc_action = tau + proc_action

        self._robot.ApplyAction(proc_action)
        self._pybullet_client.stepSimulation()
        self._sim_step_counter += 1
        self._last_action = self._robot._applied_motor_torque
        #self._last_action_rl = action
        if self._is_render:
          time.sleep(0.001)
          self._render_step_helper()

        reward, terminate = self._reward_and_terminate()
        # done = terminate or self.get_sim_time() > self._MAX_EP_LEN # also have max time
        done = self.get_sim_time() > self._traj_task.get_duration() + 0 # also have max time
        total_reward += reward

      final_estimate = self._get_terminal_reward()
      #reward = total_reward
      reward = final_estimate

      if self._land_and_settle:
        for i in range(1000):
          # continue applying control for some duration
          q_des, q, dq_des, dq, tau = self._get_default_imitation_action()
          proc_action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
          foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
          tau = self._opt_traj_cartesian_pd_controller(foot_pos, foot_vel, foot_force)
          proc_action = tau + proc_action

          self._robot.ApplyAction(proc_action)
          self._pybullet_client.stepSimulation()
          self._sim_step_counter += 1
          self._last_action = self._robot._applied_motor_torque
          #self._last_action_rl = action
          if self._is_render:
            time.sleep(0.001)
            self._render_step_helper()


      return np.array(self._noisy_observation()), reward, done, {}

    raise ValueError('SHOULD NEVER GET HERE', self.get_sim_time())

  def step(self, action):
    """ Step forward the simulation, given the action. """
    if self._task_env == "FULL_TRAJ":
      #print('action', action)
      return self.step_full_traj(action)

    if self._is_render:
      self._render_step_helper()
    
    """
    TODO: should the action be additive to whatever we found with PD control?
    # action should be based on where we want to be at next time step?
    """
    curr_act = action.copy()
    if self._enable_action_filter:
      curr_act = self._action_filter(curr_act)
    #print(self._action_repeat, self._sim_step_counter, self._time_step)
    for _ in range(self._action_repeat): # 100

      # if self._enable_action_filter:
      #   curr_act = self._action_filter(curr_act)
      # only scale action if it's a gym interface
      if self._GAS:
        proc_action = self._transform_action_GAS(curr_act)
      elif self._isRLGymInterface:
        proc_action = self._transform_action_to_motor_command(curr_act)
      else:
        proc_action = curr_act
      # if self._enable_action_filter:
      #   proc_action = self._action_filter(proc_action)
      self._robot.ApplyAction(proc_action)
      self._pybullet_client.stepSimulation()
      self._sim_step_counter += 1
      self._last_action = self._robot._applied_motor_torque
      
      if self._is_render:
        time.sleep(0.001)
        self._render_step_helper()


    # check if off the ground, if so then sim until end of trajectory open loop
    #if self.get_sim_time() > FLIGHT_TIME_START: # "PREJUMP_ONLY" in self._task_mode and
    _,_,_,feetInContactBool = self._robot.GetContactInfo()
    if "PREJUMP_ONLY" in self._task_mode and self.get_sim_time() > FLIGHT_TIME_START : # and sum(feetInContactBool) == 0
      done = False
      total_reward = 0
      while self.get_sim_time() > FLIGHT_TIME_START and not done:
        #self._transform_action_to_motor_command(np.zeros(self._action_dim))
        q_des, q, dq_des, dq, tau = self._get_default_imitation_action()
        proc_action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)

        if "CARTESIAN" in self._task_mode:
          # if cartesian in name 
          # joint PD w torque
          #tau_ff = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
          # Cartesian PD
          foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
          tau = self._opt_traj_cartesian_pd_controller(foot_pos, foot_vel, foot_force)
          proc_action = tau + proc_action

        self._robot.ApplyAction(proc_action)
        self._pybullet_client.stepSimulation()
        self._sim_step_counter += 1
        self._last_action = self._robot._applied_motor_torque
        #self._last_action_rl = action
        if self._is_render:
          time.sleep(0.001)
          self._render_step_helper()

        reward, terminate = self._reward_and_terminate()
        # done = terminate or self.get_sim_time() > self._MAX_EP_LEN # also have max time
        done = self.get_sim_time() > self._traj_task.get_duration() + 0 # also have max time
        total_reward += reward

      final_estimate = self._get_terminal_reward()
      reward = total_reward

      return np.array(self._noisy_observation()), reward, done, {}

    self._last_action_rl = action
    self._env_step_counter += 1
    #reward = self._reward()
    #self._last_action = self._robot._applied_motor_torque
    reward, terminate = self._reward_and_terminate()
    done = terminate or self.get_sim_time() > self._MAX_EP_LEN # also have max time
    #done = self.get_sim_time() > self._MAX_EP_LEN 

    # reward only if at end of trajectory, otherwise 0
    if self._sparse_rewards:
      reward = 0

    # if at end of trajectory, give extra reward on how close we are to desired
    if self.get_sim_time() > self._traj_task.get_duration():
      reward = self._get_terminal_reward()



    if done and self._is_render:
      time.sleep(0.5)
      self._render_step_helper()
    return np.array(self._noisy_observation()), reward, done, {}

  ######################################################################################
  # Termination and reward
  ######################################################################################
  def _get_terminal_reward(self):
    curr_time = self.get_sim_time()
    # desired trajectory state at this time
    traj_base_pos, traj_base_orn, traj_base_linvel, \
    traj_base_angvel, traj_joint_pos, traj_joint_vel = self.get_traj_state_at_time(curr_time)
    # current robot state
    curr_base_pos, curr_base_orn, curr_base_linvel, \
    curr_base_angvel, curr_joint_pos, curr_joint_vel = self.get_robot_state()
    # quaternions 
    traj_base_orn = np.array(self._pybullet_client.getQuaternionFromEuler([0,traj_base_orn[1],0]))
    curr_base_orn = np.array(self._robot.GetBaseOrientation())

    if np.linalg.norm(traj_base_orn - curr_base_orn) > np.linalg.norm(traj_base_orn + curr_base_orn):
      # means we should take negative of the orn
      curr_base_orn = -curr_base_orn

    # just do pitch for now?
    # traj_base_orn = traj_base_orn[1]
    # curr_base_orn = curr_base_orn[1]

    reward = 100 * np.linalg.norm( traj_base_pos - curr_base_pos ) \
           + 100 * np.linalg.norm( traj_base_orn - curr_base_orn ) 
    reward = 100 - reward # survival bonus

    # try exp reward
    # reward = np.exp( -1.5*np.sum( ( traj_base_pos - curr_base_pos )**2 ) ) \
    #        + np.exp( -50*np.sum( ( traj_base_orn - curr_base_orn )**2 ) )

    print('final reward', reward)

    print('traj ', np.array(traj_base_pos),traj_base_orn)
    print('curr', np.array(curr_base_pos),curr_base_orn)
    # print('-'*30)
    # print('RPY ', np.array(self._robot.GetBaseOrientationRollPitchYaw()))
    # print('quat', np.array(self._robot.GetBaseOrientation()))
    # print('comp', np.array(self._pybullet_client.getEulerFromQuaternion(self._robot.GetBaseOrientation())))

    return reward

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
    curr_time = self.get_sim_time()

    # desired trajectory state at this time
    traj_base_pos, traj_base_orn, traj_base_linvel, \
    traj_base_angvel, traj_joint_pos, traj_joint_vel = self.get_traj_state_at_time(curr_time)

    # current robot state
    curr_base_pos, curr_base_orn, curr_base_linvel, \
    curr_base_angvel, curr_joint_pos, curr_joint_vel = self.get_robot_state()

    # quaternions 
    traj_base_orn = np.array(self._pybullet_client.getQuaternionFromEuler([0,traj_base_orn[1],0]))
    curr_base_orn = np.array(self._robot.GetBaseOrientation())
    # traj_base_orn = np.array([traj_base_orn[1]])
    # curr_base_orn = np.array([curr_base_orn[1]])

    if np.linalg.norm(traj_base_orn - curr_base_orn) > np.linalg.norm(traj_base_orn + curr_base_orn):
      # means we should take negative of the orn
      curr_base_orn = -curr_base_orn

    self._base_pos_weight = 1
    self._base_orn_weight = .5 #10 #.5
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
    # print('traj ')
    # print(np.array(traj_base_pos),traj_base_orn)
    # print('curr')
    # print(np.array(curr_base_pos),curr_base_orn)
    # print('orientation')
    # print('RPY ', np.array(self._robot.GetBaseOrientationRollPitchYaw()))
    # print('quat', np.array(self._robot.GetBaseOrientation()))
    # print('comp', np.array(self._pybullet_client.getEulerFromQuaternion(self._robot.GetBaseOrientation())))
    # print('-'*30)

    # always strange..
    # terminate if diverge too much from trajectory
    #if np.linalg.norm( traj_base_pos - curr_base_pos) > .5: #1:
    if np.linalg.norm( np.concatenate((traj_base_pos,traj_base_orn)) \
             - np.concatenate((curr_base_pos,curr_base_orn)) ) > .5: #1:
    # if np.linalg.norm( np.concatenate((traj_base_pos,[traj_base_orn[1]])) \
    #      - np.concatenate((curr_base_pos,[curr_base_orn[1]])) ) > .2: #1:
      terminate = True
      #print('*'*80)
      #terminate = False


    # if curr_time > self._traj_task.get_duration():
    #   # different reward
    #   #rewrd = 1 - 10*self._base_pos_weight * np.linalg.norm( traj_base_pos - curr_base_pos )
    #   terminate = False

    # terminate if trajectory is over, add 1 second of settling time
    if curr_time > self._traj_task.get_duration() + 0:
      terminate = True
      #print('final base pos', self._robot.GetBasePosition())
      #sys.exit()



    return reward, terminate
  ######################################################################################
  # Observation
  ######################################################################################
  def get_objectives(self):
    return self._objectives

  def _get_observation(self):
    """Get observation for imitation task.
    Concatenate current robot state with next desired observation(s) (next state(s) in traj).
    -TODO: add input to determine how many future trajectory snapshots to include in observation 
    (see google paper, which has something like 1 dt ahead, 5 dt ahead, 0.33 seconds ahead, 1 second ahead) 
    """
    # [now, 0.1, 0.5, 1] seconds ahead?
    curr_time = self.get_sim_time()
    traj_dt = self._traj_task.dt
    # get times at which to query (should be option flag?)
    # traj_query_times = curr_time+traj_dt + np.array([0, 0.1, 0.5, 1])

    if self._task_env == "FULL_TRAJ":
      # just get last spot
      traj_query_times = np.array([self._traj_task.get_duration()])
    else:
      traj_query_times = curr_time+traj_dt + np.array([0, 0.05, 0.2, self._traj_task.get_duration()])
    traj_duration = self._traj_task.get_duration()
    # times at which to query opt traj data
    #obs_times = min(curr_time+traj_dt, traj_duration)
    obs_times = np.minimum(traj_query_times, traj_duration)
    # print("obs_times: ", obs_times)
    # current robot state
    if self._observation_space_mode == "MOTION_IMITATION_EXTENDED_CONTACTS":
      robot_state = self._robot.GetExtendedObservation()
    elif self._observation_space_mode == "MOTION_IMITATION_EXTENDED2_CONTACTS":
      robot_state = self._robot.GetExtendedObservation2()
      print("robot_state1:", len(robot_state)) # 73
      # print(type(robot_state)) # list type
    elif self._observation_space_mode == "MOTION_IMITATION_EXTENDED2_CONTACTS_HISTORY":
      robot_state = self._robot.GetExtendedObservation2_History() # CHUONG, get history
      # print("robot_state_size:", robot_state.shape)
    else:
      raise ValueError('wrong obs mode')
    # next desired traj state (if there are multiple, will need to call once for each)
    # traj_desired_state = self._traj_task.get_state_at_time(obs_times)
    traj_desired_states = np.array([])
    for obs_time in obs_times:
      traj_desired_states = np.concatenate((traj_desired_states, 
                                            self._traj_task.get_state_at_time(obs_time)))
      # print('traj_desired_states shape:', traj_desired_states.shape)

    self._observation = np.hstack((robot_state, np.tile(traj_desired_states, self._sensory_hist_len)))

    # print("obs_1 size:", self._observation.shape)

    # if self._observation_space_mode == "MOTION_IMITATION_EXTENDED_CONTACTS":
    if "CONTACTS" in self._observation_space_mode:
      _, _, _, feetInContactBool = self._robot.GetContactInfo()
      # print("feetInContactBool_type", feetInContactBool.type)
      self._observation = np.hstack((self._observation, feetInContactBool*self._sensory_hist_len))
    #   print('in get observation ', self._observation.shape)
    # print('self obs mode', self._observation_space_mode)
    # print('in get observation  but didnt get called', self._observation.shape)

    print("obs_2 size:", self._observation.shape)
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    # if self._observation_noise_stdev > 0:
    #   raise NotImplementedError
    #   observation += (
    #           np.random.normal(scale=self._observation_noise_stdev, size=observation.shape) *
    #           self._robot.GetObservationUpperBound())
    return observation

    ##############################################################################
    ## Imitation helpers
    ##############################################################################

  def get_traj_state_at_time(self, t):
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
    base_pos = self._traj_task.get_base_pos_from_state(traj_state)  # ,env2D=self._env2D)
    base_orn = self._traj_task.get_base_orn_from_state(traj_state)  # ,env2D=self._env2D)
    base_linvel = self._traj_task.get_base_linvel_from_state(traj_state)  # ,env2D=self._env2D)
    base_angvel = self._traj_task.get_base_angvel_from_state(traj_state)  # ,env2D=self._env2D)

    joint_pos = self._traj_task.get_joint_pos_from_state(traj_state)  # ,env2D=self._env2D)
    joint_vel = self._traj_task.get_joint_vel_from_state(traj_state)  # ,env2D=self._env2D)

    return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

  def get_traj_state_at_index(self, idx):
    """ Get the traj state at index idx.
    Return:
    -base_pos
    -base_orn
    -base_linvel
    -base_angvel
    -joint_pos
    -joint_vel
    """
    traj_state = self._traj_task.get_state_at_index(idx)
    base_pos = self._traj_task.get_base_pos_from_state(traj_state)  # ,env2D=self._env2D)
    base_orn = self._traj_task.get_base_orn_from_state(traj_state)  # ,env2D=self._env2D)
    base_linvel = self._traj_task.get_base_linvel_from_state(traj_state)  # ,env2D=self._env2D)
    base_angvel = self._traj_task.get_base_angvel_from_state(traj_state)  # ,env2D=self._env2D)

    joint_pos = self._traj_task.get_joint_pos_from_state(traj_state)  # ,env2D=self._env2D)
    joint_vel = self._traj_task.get_joint_vel_from_state(traj_state)  # ,env2D=self._env2D)

    return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

  def get_cartesian_traj_state_at_time(self, t):
    """ Get the Cartesian related info at time t. """
    traj_state = self._traj_task.get_state_at_time(t)
    foot_pos = self._traj_task.get_foot_pos_from_state(traj_state)
    foot_vel = self._traj_task.get_foot_vel_from_state(traj_state)
    foot_force = self._traj_task.get_foot_force_from_state(traj_state)

    return foot_pos, foot_vel, foot_force

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
    base_pos = self._robot.GetBasePosition()
    base_orn = self._robot.GetBaseOrientationRollPitchYaw()
    # print('new base euler', base_orn)
    base_orn = self.convertAnglesToPlusMinusPi2(base_orn)
    # print('last base euler', self._last_base_euler)
    self._last_base_euler = base_orn
    # print('curr base euler', self._last_base_euler)

    base_linvel = self._robot.GetBaseLinearVelocity()
    # base_angvel = self._robot.GetBaseAngularVelocity()
    base_angvel = self._robot.GetTrueBaseRollPitchYawRate()
    # joint_pos, joint_vel = np.split(self._robot.GetJointState(), 2)
    joint_pos = self._robot.GetMotorAngles()
    joint_vel = self._robot.GetMotorVelocities()

    return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

  def convertAnglesToPlusMinusPi2(self, angles):
    """Converts angles to +/- pi/2

    Some lingering bugs in wraparound, and pitch should be double checked as well.
    May be better off keeping everything in quaternions

    """
    # new_angles = []
    # for a in angles:
    #   while a > np.pi/2:
    #     a -= np.pi
    #   while a < -np.pi/2:
    #     a += np.pi
    #   new_angles.append(a)
    # new_angles = np.zeros(len(angles))

    # orginal w/ pitch bug

    # new_angles = angles.copy()
    # for i, a in enumerate(angles):
    #   if abs(self._last_base_euler[i] - a) > np.pi/2:
    #     if self._last_base_euler[i] > new_angles[i]:
    #       while self._last_base_euler[i] > new_angles[i] and abs(self._last_base_euler[i] - new_angles[i]) > np.pi/2:
    #         # skipped across 0 to around pi
    #         #print('pre  update: self._last_base_euler[i]', self._last_base_euler[i], 'new_angles[i]',  new_angles[i])
    #         new_angles[i] += np.pi
    #         #print('post update: self._last_base_euler[i]', self._last_base_euler[i], 'new_angles[i]',  new_angles[i])
    #     elif self._last_base_euler[i] < new_angles[i]:
    #       while self._last_base_euler[i] < new_angles[i] and abs(self._last_base_euler[i] - new_angles[i]) > np.pi/2:
    #         #print('pre  update: self._last_base_euler[i]', self._last_base_euler[i], 'new_angles[i]',  new_angles[i])
    #         new_angles[i] -= np.pi
    #         #print('post update: self._last_base_euler[i]', self._last_base_euler[i], 'new_angles[i]',  new_angles[i])
    #   else:
    #     new_angles[i] = a
    # return np.array(new_angles)
    base_orn = self._robot.GetBaseOrientation()
    new_angles = angles.copy()
    for i, a in zip([0, 2, 1], [angles[0], angles[2], angles[1]]):  # enumerate(angles):
      if i == 1:
        # check if the value of transforming pitch back to quaternion space is same
        # if not, check value, and can subtract it from pi or -pi
        # print('current quaternion', base_orn)
        # print('pitch from conversion', a)
        new_base_orn = self._pybullet_client.getQuaternionFromEuler([new_angles[0], a, new_angles[2]])
        # print('pitch to quaternion', env._pybullet_client.getQuaternionFromEuler([angles[0], a, angles[2]]))
        if not np.allclose(base_orn, new_base_orn, rtol=1e-3, atol=1e-3):
          # if np.linalg.norm(np.array(base_orn) - np.array(new_base_orn)) > 5e-7:
          # if abs(base_orn[1] - new_base_orn[1]) > 1e-4:
          if a <= 0:
            new_angles[i] = -np.pi - a
          else:  # should never get here
            new_angles[i] = np.pi - a
        else:
          new_angles[i] = a

      elif abs(self._last_base_euler[i] - a) > np.pi / 2:
        # print('i', i)
        if self._last_base_euler[i] > new_angles[i]:
          while self._last_base_euler[i] > new_angles[i] and abs(self._last_base_euler[i] - new_angles[i]) > np.pi / 2:
            # skipped across 0 to around pi
            # print('pre  update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
            new_angles[i] += np.pi
            # print('post update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
        elif self._last_base_euler[i] < new_angles[i]:
          while self._last_base_euler[i] < new_angles[i] and abs(self._last_base_euler[i] - new_angles[i]) > np.pi / 2:
            # print('pre  update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
            new_angles[i] -= np.pi
            # print('post update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
      else:
        new_angles[i] = a
    return np.array(new_angles)

    ######################################################################################
    # Render and misc
    ######################################################################################

  def set_env_randomizer(self, env_randomizer):
    self._env_randomizer = env_randomizer

    # def recordVideoHelper(self):
    #   """ Helper to record video, if not already, or end and start a new one """
    #   # If no ID, this is the first video, so make a directory and start logging
    #   if self.videoLogID == None:
    #     #directoryName = VIDEO_LOG_DIRECTORY + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")
    #     directoryName = VIDEO_LOG_DIRECTORY
    #     assert isinstance(directoryName, str)
    #     os.makedirs(directoryName, exist_ok=True)
    #     self.videoDirectory = directoryName
    #   else:
    #     # stop recording and record a new one
    #     self.stopRecordingVideo()

    #   # name
    #   traj_name = self.opt_traj_filenames[self.curr_opt_traj]
    #   output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + traj_name[:-4] + ".MP4"
    #   logID = self.startRecordingVideo(output_video_filename)
    #   self.videoLogID = logID

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
