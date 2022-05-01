"""This file implements the gym environment of a quadruped. 
Subclass as necessary. 

TODO:
decouple action space from this file and from quadruped.py, make a new class ActionSpace.py

"""
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import time, datetime
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
#import os
import random
random.seed(10)
from collections import deque 

import usc_learning.envs.pyb_data as pyb_data
from usc_learning.envs.my_minitaur import my_minitaur_env_randomizer
import usc_learning.envs.quadruped_master.quadruped as quadruped
from usc_learning.utils.foot_trajectory_generator import FootTrajectoryGenerator
from usc_learning.utils.bezier_gait import BezierGait
from pkg_resources import parse_version

ROBOT_FLAG = 1
# 0 aliengo
# 1 A1
# 2 Laikago

if ROBOT_FLAG == 0:
  import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
elif ROBOT_FLAG == 1:
  import usc_learning.envs.quadruped_master.configs_a1 as robot_config
elif ROBOT_FLAG == 2:
  import usc_learning.envs.quadruped_master.configs_laikago as robot_config
else:
  raise ValueError('This robot does not exist.')


ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
VIDEO_LOG_DIRECTORY = 'videos/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")

DEBUG_FLAG = False

# Current observation options are 
OBSERVATION_SPACES = ["DEFAULT", "DEFAULT_CONTACTS", "EXTENDED", "EXTENDED2", "EXTENDED2VISION"]
# DEFAULT:          motor angles, velocities, torques; body orientation
# DEFAULT_CONTACTS: DEFAULT + body height z + body linear velocities, feet contact booleans 
# EXTENDED:         DEFAULT + body linear and angular velocities (used for jumping task)
# EXTENDED2:        EXTENDED + foot pos/vel

""" Implemented task environemnts """
TASK_ENVS = [ "DEFAULT",          # original reward function from minitaur env
              "FWD_LOCOMOTION",   # DEFAULT + survival reward
              "FWD_SET_VELOCITY", # learn fwd locomotion at particular velocity (default 1 m/s)
              "STANDUP",          # stay standing at particular height
              "STAYDOWN",         # not implemented for this env
              "GOAL",             # get to goal location as fast as possible 
              ]

# DES_VEL_LOW = 1.
# DES_VEL_HIGH = 2.

# DES_VEL_LOW = 0.3
# DES_VEL_HIGH = 1.2

# for flat terrain w impedance control 
DES_VEL_LOW = 1
DES_VEL_HIGH = 6

# Available control modes, not all implemented
# better format? { XYZ, 3, F, V, K } if contains any of these, add
CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", 
                  "PD", "PD_NO_HIPS", "IK", #"FORCE", 
                  "IMPEDANCE",          # agent chooses foot xyz and Fn (16 values)
                  "IMPEDANCE_POS_ONLY", # agent chooses foot xyz only
                  "IMPEDANCE_POS_ONLY_SPHERICAL",    # agent chooses spherical coordinates, convert to foot positions
                  "IMPEDANCE_POS_ONLY_SPHERICAL_IK", # agent chooses spherical coordinates, joint PD control as well
                  "IMPEDANCE_POS_VEL_ONLY", # agent chooses foot xyz and foot velocity only
                  "IMPEDANCE_XZ_ONLY",  # choose xz only
                  "IMPEDANCE_NO_Y",     # agent chooses foot xz and Fn  (note this is also 12 values like above, careful)
                  #"IMPEDANCE_FORCE",   # agent chooses Fn only, xyz are under hips (4 values) (i.e. to test standing)
                  "IMPEDANCE_GAINS",    # agent chooses foot xyz, Fn, kpCartesian, kdCartesian (22 values)
                  #"IMPEDANCE_CONTACTS",# TODO: apply Fn only in contact (not really needed?)
                  "IMPEDANCE_3D_FORCE", # agent chooses foot xyz and 3D foot force vector (24 values)
                  "IMPEDANCE_FULL",     # agent chooses foot xyz pos/vel and 3D foot force vector (36 values)
                  "IMPEDANCE_TROT",     # agent chooses FL/FR xyz which will be same for corresponding diagonal legs, 
                                        #   but a separate Fn for each (same diag did not work), (10 values)
                  "PMTG",               # choose frequency fi for each leg, and xy offsets for each (12 values)
                  "PMTG_XY_ONLY",       # xy offsets for each foot only (8 values)
                  "PMTG_JOINT",          # instead of impedance, use joint PD
                  "PMTG_XY_ONLY_JOINT",          # instead of impedance, use joint PD
                  # with Z equivalants
                  "PMTG_XYZ_ONLY", "PMTG_XYZ_ONLY_JOINT",
                  "PMTG_FULL", "PMTG_FULL_JOINT",
                  "BEZIER_GAIT",        # impedance space
                  "BEZIER_GAIT_JOINT", # joint space
                   ]

EPISODE_LENGTH = 10 # in seconds
MAX_FWD_VELOCITY = 6 #np.inf#10 #np.inf

class Normalizer():
  """ this ensures that the policy puts equal weight upon
      each state component.
  """

  # Normalizes the states
  def __init__(self, state_dim):
    """ Initialize state space (all zero)
    """
    self.state = np.zeros(state_dim)
    self.mean = np.zeros(state_dim)
    self.mean_diff = np.zeros(state_dim)
    self.var = np.zeros(state_dim)

  def observe(self, x):
    """ Compute running average and variance
        clip variance >0 to avoid division by zero
    """
    self.state += 1.0
    last_mean = self.mean.copy()

    # running avg
    self.mean += (x - self.mean) / self.state

    # used to compute variance
    self.mean_diff += (x - last_mean) * (x - self.mean)
    # variance
    self.var = (self.mean_diff / self.state).clip(min=1e-2)

  def normalize(self, states):
    """ subtract mean state value from current state
        and divide by standard deviation (sqrt(var))
        to normalize
    """
    state_mean = self.mean
    state_std = np.sqrt(self.var)
    return (states - state_mean) / state_std

class QuadrupedGymEnv(gym.Env):
  """The gym environment for quadrupeds {Aliengo, Laikago, A1}.

  It simulates the locomotion of a quadrupedal robot. 
  The state space, action space, and reward functions can be chosen with:
  observation_state_space, motor_control_mode, task_env
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the quadruped walks in 1000 steps and penalizes the energy
  expenditure.

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
  NUM_SUBSTEPS = 5 # originally 5

  def __init__(
      self,
      urdf_root=pybullet_data.getDataPath(), #pyb_data.getDataPath() # careful_here 
      robot_config=robot_config,
      isRLGymInterface=True, # True means running RL, should be false if running tests
      time_step=0.001,
      action_repeat=20, # 50 seems to be working, couldn't verify though
      distance_weight=2.0,
      energy_weight=0.008,#05,#0.005,#0.005, # orig 0.1
      shake_weight=0,#1e-4,
      drift_weight=0,#1e-4,
      distance_limit=float("inf"),
      observation_noise_stdev=0,
      self_collision_enabled=True,
      motor_velocity_limit=np.inf,
      pd_control_enabled=False,  #not needed to be true if accurate motor model is enabled (has its own better PD)
      #leg_model_enabled=True,
      accurate_motor_model_enabled=True,
      motor_control_mode="IMPEDANCE_POS_ONLY",
      output_mode=None,#"N_TRAJ",    # RL "action" is several trajectory points as 1 pass through NN, otherwise should be None
      output_mode_num=5,         # how many points to look at in future 
      task_env="FWD_LOCOMOTION",
      observation_space_mode="DEFAULT_CONTACTS",
      time_remaining_in_obs_space=True,
      internal_normalizer=False, # whether to normalize observations internally, works only for history length 1 for now
      observation_history_length=1,
      rl_action_high=1.0,       # action high for RL
      grow_action_space=False,  # use growing action spaces
      scale_gas_action=True,    # True: scale action range (i.e. smaller than [-1,1]), False: scale high/low limits being scaled to
      motor_overheat_protection=True,
      hard_reset=True,
      on_rack=False,
      enable_action_filter=False, # whether to filter (low pass) so actions do not change too fast
      action_filter_weight=0.2,
      render=False,
      record_video=False,
      env_randomizer=None,
      randomize_dynamics=True,
      add_terrain_noise=True, #my_minitaur_env_randomizer.MinitaurEnvRandomizer()):
      external_force_probability= 0.01, # probability of receiving external force at body (TODO: tune magnitude)
      test_mass=None,
      **kwargs): # any extras from legacy
    """Initialize the quadruped gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      robot_config: The robot config file, can be Aliengo, Laikago, A1.
      isRLGymInterface: If the gym environment is being run as RL or not. Affects
        if the actions should be scaled.
      time_step: Sim time step. WILL BE CHANGED BY NUM_SUBSTEPS.
      action_repeat: The number of simulation steps before actions are applied.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space. Should be TRUE
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_control_mode: Whether to use torque control, PD, control, etc.
      task_env: Task trying to learn (fwd locomotion, standup, etc.)
      observation_space_mode: Check quadruped.py, default, contacts, extended, etc.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in quadruped.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the quadruped back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the quadruped on rack. This is only used to debug
        the walking gait. In this mode, the quadruped's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      record_video: Whether to record a video of each trial.
      env_randomizer: TODO: An EnvRandomizer to randomize the physical properties
        during reset().
      add_terrain_noise: add random boxes 
    """
    self._robot_config = robot_config
    self._isRLGymInterface = isRLGymInterface
    self._time_step = time_step
    self._action_repeat = action_repeat
    self._num_bullet_solver_iterations = 60 #300
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []
    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._is_render = render
    self._is_record_video = record_video
    self._last_base_position = [0, 0, 0]
    self._distance_weight = distance_weight
    self._energy_weight = energy_weight
    self._drift_weight = drift_weight
    self._shake_weight = shake_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    
    self._pd_control_enabled = pd_control_enabled
    #self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._motor_control_mode = motor_control_mode
    self._TASK_ENV = task_env
    self._observation_space_mode = observation_space_mode
    self._normalize_obs = internal_normalizer
    self._test_mass = test_mass

    # action space
    self._GAS = grow_action_space
    self._rl_action_bound = rl_action_high
    self._action_bound = 1.0
    self._scale_GAS_action = scale_gas_action
    if "PMTG" in motor_control_mode:
      self.setupFTG() 
    if "BEZIER" in motor_control_mode:
      self.setupBezier() 

    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack

    self._hard_reset = True
    self._last_frame_time = 0.0
    print("urdf_root=" + self._urdf_root)
    self._env_randomizer = env_randomizer
    self._randomize_dynamics = randomize_dynamics
    self._terrain_noise = add_terrain_noise

    self._enable_action_filter = enable_action_filter
    self._action_filter_weight = action_filter_weight
    self._output_mode = output_mode
    self._output_mode_num = output_mode_num
    self._external_force_probability = external_force_probability
    # history for obs
    self._observation_history_length = observation_history_length
    self._observation_history = deque(maxlen=observation_history_length)
    self._MAX_EP_LEN = EPISODE_LENGTH # max sim time in seconds, arbitrary
    self._time_remaining_in_obs_space = time_remaining_in_obs_space
    #

    self.setupActionSpace()

    # TODO: fix, this is not clean
    # PD control needs smaller time step for stability.
    # if pd_control_enabled or accurate_motor_model_enabled:
    #   self._time_step /= self.NUM_SUBSTEPS
    #   self._num_bullet_solver_iterations /= self.NUM_SUBSTEPS
    #   self._action_repeat *= self.NUM_SUBSTEPS

    # if not isRLGymInterface:
    #   self._time_step = time_step
    #   self._action_repeat = action_repeat
    #   self._num_bullet_solver_iterations = 300

    if self._is_render:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bc.BulletClient()
    self._configure_visualizer()

    self.videoLogID = None
    self.seed()
    self.reset()
    self.setupObsSpace()
    self._hard_reset = hard_reset  # This assignment need to be after reset()

    
  def setupFTG(self):
    self._ftg = FootTrajectoryGenerator(T=0.5,max_foot_height=0.1)

  def setupBezier(self):
    self._bzg = BezierGait(dt=self._time_step)

  def setupObsSpace(self):
    """Set up observation and action spaces for RL. """
    # Observation space
    if self._observation_space_mode == "DEFAULT":
      observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
      observation_low =  (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)
    elif self._observation_space_mode == "DEFAULT_CONTACTS":
      observation_high = (self._robot.GetObservationContactsUpperBound() + OBSERVATION_EPS)
      observation_low =  (self._robot.GetObservationContactsLowerBound() - OBSERVATION_EPS)
    elif self._observation_space_mode == "EXTENDED":
      observation_high = (self._robot.GetExtendedObservationUpperBound() + OBSERVATION_EPS)
      observation_low =  (self._robot.GetExtendedObservationLowerBound() - OBSERVATION_EPS)
    elif self._observation_space_mode == "EXTENDED2":
      observation_high = (self._robot.GetExtendedObservation2UpperBound() + OBSERVATION_EPS)
      observation_low =  (self._robot.GetExtendedObservation2LowerBound() - OBSERVATION_EPS)
    elif self._observation_space_mode == "EXTENDED2VISION":
      observation_high = (self._robot.GetExtendedObservation2VisionUpperBound() + OBSERVATION_EPS)
      observation_low =  (self._robot.GetExtendedObservation2VisionLowerBound() - OBSERVATION_EPS)
    else:
      raise ValueError("observation space not defined or not intended")

    if self._TASK_ENV == "FWD_SET_VELOCITY":
      observation_high = np.append( observation_high, DES_VEL_HIGH + OBSERVATION_EPS)
      observation_low  = np.append( observation_low,  0  - OBSERVATION_EPS)

    if "PMTG" in self._motor_control_mode:
      # add phases
      observation_high = np.append( observation_high, np.array([2*np.pi]*4) + OBSERVATION_EPS ) 
      observation_low  = np.append( observation_low,  np.array([0]*4) - OBSERVATION_EPS ) 

    if self._observation_history_length > 1:

      observation_high = np.tile(observation_high, self._observation_history_length)
      observation_low =  np.tile(observation_low , self._observation_history_length)
      # print('==================================================================================')
      # print('obs high', observation_high)

    if self._time_remaining_in_obs_space:
      observation_high = np.append( observation_high, self._MAX_EP_LEN + OBSERVATION_EPS)
      observation_low  = np.append( observation_low,  0  - OBSERVATION_EPS)

    self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
    if self._normalize_obs:
      self.normalizer = Normalizer(len(observation_low))


  def setupActionSpace(self):
    # Action space
    if self._motor_control_mode in ["IMPEDANCE_XZ_ONLY", "PMTG_XY_ONLY", "PMTG_XY_ONLY_JOINT"]:
      action_dim = 8
    elif self._motor_control_mode in ["IMPEDANCE_TROT"]:
      action_dim = 6
    elif self._motor_control_mode in ["PD","TORQUE", "IK", "IMPEDANCE_POS_ONLY", "IMPEDANCE_POS_ONLY_IK",
                                      "IMPEDANCE_POS_ONLY_SPHERICAL", "IMPEDANCE_POS_ONLY_SPHERICAL_IK",
                                      "IMPEDANCE_NO_Y", "PMTG", "PMTG_JOINT"]:
      action_dim = 12
    elif self._motor_control_mode in ["IMPEDANCE_POS_GAINS"]:
      action_dim = 14
    elif self._motor_control_mode in ["PMTG_XYZ_ONLY", "PMTG_XYZ_ONLY_JOINT"]:
      action_dim = 12
    elif self._motor_control_mode in ["PMTG_FULL", "PMTG_FULL_JOINT"]:
      action_dim = 16
    elif self._motor_control_mode in ["BEZIER_GAIT", "BEZIER_GAIT_JOINT"]:
      # just for now... what should this really be?
      action_dim = 12 
    elif self._motor_control_mode in ["IMPEDANCE"]:
      action_dim = 16 # (xyz pos + Fn)*4
    # elif self._motor_control_mode in ["IMPEDANCE_FORCE"]:
    #   action_dim = 4 # Fn*4
    elif self._motor_control_mode in ["IMPEDANCE_GAINS"]:
      action_dim = 22 # (xyz pos + Fn)*4 + 3 kp_gains + 3 kd_gains
    elif self._motor_control_mode in ["IMPEDANCE_3D_FORCE", "IMPEDANCE_POS_VEL_ONLY"]:
      action_dim = 24 # (xyz pos + 3D contact force)*4  
    elif self._motor_control_mode in ["IMPEDANCE_FULL"]:
      action_dim = 36 # (xyz pos/vel + 3D contact force)*4  
    else:
      raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")

    if self._output_mode == "N_TRAJ":
      # update action_dim based on full N traj outputs 
      self._action_dim_orig_action_space = action_dim
      action_dim = action_dim * self._output_mode_num

    if self._GAS: # likely in lower range than [-1,1]
      if self._scale_GAS_action:
        action_high = np.array([self._rl_action_bound] * action_dim)
      else:
        action_high = np.array([self._action_bound] * action_dim)
    else:
      action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

    self._action_dim = action_dim
    self._last_action_rl = np.zeros(self._action_dim)

    

  ######################################################################################
  # Reset
  ######################################################################################
  def reset(self):
    mu_min = 0.5
    if self._hard_reset:
      #print('HARD RESET')
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      # if self._terrain_noise:
      #   self.make_rough_terrain(heightPerturbationRange=0.05)
      # else:
      #   self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root, 
                                                  basePosition=[80,0,0], # to extend available running space (shift)
                                                  )
      self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._pybullet_client.setGravity(0, 0, -9.8)
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

      if self._terrain_noise:
        ground_mu_k = mu_min+(1-mu_min)*np.random.random()
        self._ground_mu_k = ground_mu_k
        self.add_random_boxes()
        # self.add_random_boxes_horizontal()
      #self.setupObsSpace()
    else:
      self._robot.Reset(reload_urdf=False)

    if self._env_randomizer is not None:
      raise ValueError('Not implemented randomizer yet.')
      self._env_randomizer.randomize_env(self)

    self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=1)
    if self._randomize_dynamics and self._test_mass is None:
      if np.random.random() < 0.6:#0.1:#0.6: # don't always put in a mass..
        # self._robot.RandomizePhysicalParams(mass_percent_change=0.1,base_only=False) # base_only=True
        self._robot.RandomizePhysicalParams(mass_percent_change=0.2,base_only=False) # base_only=True
      if np.random.random() < 0.8:
        self._add_base_mass_offset()
    elif self._test_mass is not None:
      self._add_base_mass_offset(spec_mass=self._test_mass[0], spec_location=self._test_mass[1:])

    # if self._terrain_noise:
    # set random lateral friction for plane (also rotate?)
    #print(self._pybullet_client.getDynamicsInfo(self.plane, -1))
    #mu_min = 0.6
    ground_mu_k = mu_min+(1-mu_min)*np.random.random()
    self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
    if self._is_render:
      print('ground friction coefficient', ground_mu_k)

    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    self._terminate = False
    if self._is_render:
      self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])

    self._settle_robot()

    self._last_action_rl = np.zeros(self._action_dim)
    self._robot._last_action_rl = self._last_action_rl
    self._des_velocity = DES_VEL_LOW + (DES_VEL_HIGH - DES_VEL_LOW) * np.random.random()
    if self._is_render and self._TASK_ENV == "FWD_SET_VELOCITY":
      print('des velocity is', self._des_velocity)

    for i in range(self._observation_history_length):
      # populate observation history
      self._get_observation()
    # reset FTG
    if "PMTG" in self._motor_control_mode:
      self.setupFTG() 
    if "BEZIER" in self._motor_control_mode:
      self.setupBezier() 
    # print(self._print_robot_states())

    if self._is_record_video:
      self.recordVideoHelper()
    return self._noisy_observation()

  def _settle_robot(self):
    """ Settle robot and add noise to init configuration. """
    # change to PD control mode to set initial position, then set back..
    # TODO: make this cleaner
    tmp_save_motor_control_mode_ENV = self._motor_control_mode
    tmp_save_motor_control_mode_ROB = self._robot._motor_control_mode
    self._motor_control_mode = "PD"
    self._robot._motor_control_mode = "PD"
    try:
      tmp_save_motor_control_mode_MOT = self._robot._motor_model._motor_control_mode
      self._robot._motor_model._motor_control_mode = "PD"
    except:
      pass

    #print('before', self._robot.GetBaseAngularVelocity())
    if self._is_render:
      time.sleep(0.2)
    for _ in range(200):
      #if self._pd_control_enabled or self._accurate_motor_model_enabled:
      self._robot.ApplyAction(self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS)
      if self._is_render:
        time.sleep(0.01)
        #numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = self._robot.GetContactInfo()
        #print('all normals',feetNormalForces, sum(feetNormalForces))
        # print('pos', self._robot.GetBasePosition())
        # print([self._robot.GetLocalEEPos(i) for i in range(4)])
        # print(self._robot_config.NOMINAL_IMPEDANCE_POS, self._robot_config.NOMINAL_IMPEDANCE_FN)
      self._pybullet_client.stepSimulation()

    # set some init noise
    def _noise(n,delta):
      return delta * np.random.random(n) - delta/2
    noise_offset = 0.1
    base_pos = self._robot.GetBasePosition()
    self._robot.ResetBasePositionAndOrientation(base_pos,_noise(3,noise_offset))
    self._robot.ResetBaseVelocity(_noise(3,noise_offset), 
                                  _noise(3,noise_offset))
    self._robot.ApplyAction(self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS\
                            + _noise(12,0.1))
    self._pybullet_client.stepSimulation()
    
    # set control mode back
    self._motor_control_mode = tmp_save_motor_control_mode_ENV
    self._robot._motor_control_mode = tmp_save_motor_control_mode_ROB
    try:
      self._robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
    except:
      pass

  def _add_base_mass_offset(self, spec_mass=None, spec_location=None):
    quad_base = np.array(self._robot.GetBasePosition())
    quad_ID = self._robot.quadruped

    # offset_low = np.array([-0.15, -0.02, -0.05])
    # offset_upp = np.array([ 0.15,  0.02,  0.05])
    offset_low = np.array([-0.15, -0.05, -0.05])
    offset_upp = np.array([ 0.15,  0.05,  0.05])
    #   block_pos_delta_base_frame = -1*np.array([-0.2, 0.1, -0.])
    if spec_location is None:
      block_pos_delta_base_frame = self.scale_rand(3,offset_low,offset_upp)
    else:
      block_pos_delta_base_frame = np.array(spec_location)
    if spec_mass is None:
      # base_mass = 8*np.random.random()
      # base_mass = 15*np.random.random()
      # base_mass = 12*np.random.random()
      base_mass = 10*np.random.random()
    else:
      base_mass = spec_mass
    if self._is_render:
      print('=========================== Random Mass:')
      print('Mass:', base_mass, 'location:', block_pos_delta_base_frame)

      # if rendering, also want to set the halfExtents accordingly 
      # 1 kg water is 0.001 cubic meters 
      boxSizeHalf = [(base_mass*0.001)**(1/3) / 2]*3
      #boxSizeHalf = [0.05]*3
      translationalOffset = [0,0,0.1]
    else:
      boxSizeHalf = [0.05]*3
      translationalOffset = [0]*3


    #sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=[0.05]*3)
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=boxSizeHalf, collisionFramePosition=translationalOffset)
    #orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    base_block_ID=self._pybullet_client.createMultiBody(baseMass=base_mass,
                                    baseCollisionShapeIndex = sh_colBox,
                                    basePosition = quad_base + block_pos_delta_base_frame,
                                    baseOrientation=[0,0,0,1])

    cid = self._pybullet_client.createConstraint(quad_ID, -1, base_block_ID, -1, self._pybullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)
    # disable self collision between box and each link
    for i in range(-1,self._pybullet_client.getNumJoints(quad_ID)):
      self._pybullet_client.setCollisionFilterPair(quad_ID,base_block_ID, i,-1, 0)
  ######################################################################################
  # Step 
  ######################################################################################
  def _transform_action_GAS(self,action):
    """Quick test """
    if self._motor_control_mode == "PD":
      # skewed.. 
      for i, a in enumerate(action):
        if a >= 0:
          action[i] = self._robot_config.NOMINAL_JOINT_POS[i] +  a * (self._robot_config.UPPER_ANGLE_JOINT[i] - self._robot_config.NOMINAL_JOINT_POS[i])
        else: # a is negative
          action[i] = self._robot_config.NOMINAL_JOINT_POS[i] -  a * (self._robot_config.LOWER_ANGLE_JOINT[i] - self._robot_config.NOMINAL_JOINT_POS[i])
      #action = self._robot.ScaleActionToPosRange(action)

    elif self._motor_control_mode == "IMPEDANCE_POS_ONLY":
      u_pos = np.zeros(12)
      for i, a in enumerate(action[:12]):
        if a >= 0:
          u_pos[i] = self._robot_config.NOMINAL_IMPEDANCE_POS[i] +  a * (self._robot_config.IK_POS_UPP_LEG[i] - self._robot_config.NOMINAL_IMPEDANCE_POS[i])
        else: # a is negative
          u_pos[i] = self._robot_config.NOMINAL_IMPEDANCE_POS[i] -  a * (self._robot_config.IK_POS_LOW_LEG[i] - self._robot_config.NOMINAL_IMPEDANCE_POS[i]) 
      action = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),np.zeros(12))

    elif self._motor_control_mode in ["IMPEDANCE_TROT"]:
      scale_xyz = np.array([0.2,0.05,0.1]) # these are max ofsets
      FR_xyz_offset = scale_xyz * action[0:3]
      FL_xyz_offset = scale_xyz * action[3:6]
      imp_pos = self._robot_config.NOMINAL_IMPEDANCE_POS
      u_pos = imp_pos + np.concatenate(( FR_xyz_offset, FL_xyz_offset, FL_xyz_offset, FR_xyz_offset ))
      u_force = np.zeros(12)

      # for i, a in enumerate(action[6:]):
      #   if a >= 0:
      #     u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] +  a * (self._robot_config.MAX_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i])
      #   else: # a is negative
      #     u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] -  a * (self._robot_config.MIN_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i]) 
      action = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),u_force)

    elif self._motor_control_mode in ["IMPEDANCE", "IMPEDANCE_NO_Y", "IMPEDANCE_XZ_ONLY", "IMPEDANCE_TROT"]:
      # scale pos, this will be symmetric
      #print(action)
      # u_pos  = self._robot_config.NOMINAL_IMPEDANCE_POS + action[:12] * ( self._robot_config.IK_POS_UPP_LEG - self._robot_config.NOMINAL_IMPEDANCE_POS )
      if self._motor_control_mode in ["IMPEDANCE_NO_Y","IMPEDANCE_XZ_ONLY"]:
        # re-create action, need a zero spaces out in pos
        tmp_a = np.zeros(16)
        for i in range(4):
          tmp_a[i*3:i*3+3] = np.array([action[2*i], 0.,action[2*i+1]])
        if self._motor_control_mode == "IMPEDANCE_NO_Y":
          tmp_a[12:] = action[8:]
        action = tmp_a

      u_pos = np.zeros(12)
      for i, a in enumerate(action[:12]):
        if a >= 0:
          u_pos[i] = self._robot_config.NOMINAL_IMPEDANCE_POS[i] +  a * (self._robot_config.IK_POS_UPP_LEG[i] - self._robot_config.NOMINAL_IMPEDANCE_POS[i])
        else: # a is negative
          u_pos[i] = self._robot_config.NOMINAL_IMPEDANCE_POS[i] -  a * (self._robot_config.IK_POS_LOW_LEG[i] - self._robot_config.NOMINAL_IMPEDANCE_POS[i]) 
          # example: with a=-. --> -50 - (-.5) * (-100 - (-50)) = -50 + . 5*(-50) - -75
      #print('pos:',u_pos,'ffwd',u_force)

      # scale forces
      u_force = np.zeros(12)
      for i, a in enumerate(action[12:]):
        if a >= 0:
          u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] +  a * (self._robot_config.MAX_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i])
        else: # a is negative
          u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] -  a * (self._robot_config.MIN_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i]) 

          # example: with a=-. --> -50 - (-.5) * (-100 - (-50)) = -50 + . 5*(-50) - -75
      #print(action)
      #print('pos:',u_pos,'ffwd',u_force)

      action = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),u_force)

    elif "PMTG" in self._motor_control_mode:
      # fine to reuse since just scaling an offset
      action = self._get_PMTG_action(action)

    elif self._motor_control_mode == "TORQUE":
      # action is some fraction of torque limits
      action = action * self._robot_config.TORQUE_LIMITS
    else:
      raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
    return action

  def _transform_action_GAS_2(self,action):
    """Scale limits being scaled to with rl_action_high,"""
    if self._motor_control_mode == "PD":
      # skewed.. 
      # action = self._robot._scale_helper(action, 
      #                                   self._rl_action_bound * self._robot_config.LOWER_ANGLE_JOINT,
      #                                   self._rl_action_bound * self._robot_config.UPPER_ANGLE_JOINT)
      # action = action + self._robot_config.NOMINAL_JOINT_POS
      #print('before scale action', action)
      for i, a in enumerate(action):
        if a >= 0:
          action[i] = self._robot_config.NOMINAL_JOINT_POS[i] +  a * self._rl_action_bound * (self._robot_config.UPPER_ANGLE_JOINT[i] - self._robot_config.NOMINAL_JOINT_POS[i])
        else: # a is negative
          action[i] = self._robot_config.NOMINAL_JOINT_POS[i] -  a * self._rl_action_bound * (self._robot_config.LOWER_ANGLE_JOINT[i] - self._robot_config.NOMINAL_JOINT_POS[i])
      #action = self._robot.ScaleActionToPosRange(action)
      #print('after scale action', action)
    elif self._motor_control_mode in ["IMPEDANCE_TROT"]:
      scale_xyz = np.array([0.2,0.05,0.1]) # these are max ofsets
      FR_xyz_offset = scale_xyz * action[0:3]
      FL_xyz_offset = scale_xyz * action[3:6]
      imp_pos = self._robot_config.NOMINAL_IMPEDANCE_POS
      u_pos = imp_pos + np.concatenate(( FR_xyz_offset, FL_xyz_offset, FL_xyz_offset, FR_xyz_offset ))
      u_force = np.zeros(12)
      # for i, a in enumerate(action[6:]):
      #   if a >= 0:
      #     u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] +  a * (self._robot_config.MAX_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i])
      #   else: # a is negative
      #     u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] -  a * (self._robot_config.MIN_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i]) 
      action = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),u_force)

    elif self._motor_control_mode in ["IMPEDANCE", "IMPEDANCE_NO_Y", "IMPEDANCE_XZ_ONLY", "IMPEDANCE_TROT"]:
      # scale pos, this will be symmetric
      #print(action)
      # u_pos  = self._robot_config.NOMINAL_IMPEDANCE_POS + action[:12] * ( self._robot_config.IK_POS_UPP_LEG - self._robot_config.NOMINAL_IMPEDANCE_POS )
      if self._motor_control_mode in ["IMPEDANCE_NO_Y","IMPEDANCE_XZ_ONLY"]:
        # re-create action, need a zero spaces out in pos
        tmp_a = np.zeros(16)
        for i in range(4):
          tmp_a[i*3:i*3+3] = np.array([action[2*i], 0.,action[2*i+1]])
        if self._motor_control_mode == "IMPEDANCE_NO_Y":
          tmp_a[12:] = action[8:]
        action = tmp_a

      u_pos = np.zeros(12)
      for i, a in enumerate(action[:12]):
        if a >= 0:
          u_pos[i] = self._robot_config.NOMINAL_IMPEDANCE_POS[i] +  a * (self._robot_config.IK_POS_UPP_LEG[i] - self._robot_config.NOMINAL_IMPEDANCE_POS[i])
        else: # a is negative
          u_pos[i] = self._robot_config.NOMINAL_IMPEDANCE_POS[i] -  a * (self._robot_config.IK_POS_LOW_LEG[i] - self._robot_config.NOMINAL_IMPEDANCE_POS[i]) 
          # example: with a=-. --> -50 - (-.5) * (-100 - (-50)) = -50 + . 5*(-50) - -75
      #print('pos:',u_pos,'ffwd',u_force)

      # scale forces
      u_force = np.zeros(12)
      for i, a in enumerate(action[12:]):
        if a >= 0:
          u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] +  a * (self._robot_config.MAX_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i])
        else: # a is negative
          u_force[2+3*i] = self._robot_config.NOMINAL_IMPEDANCE_FN[i] -  a * (self._robot_config.MIN_NORMAL_FORCE[i] - self._robot_config.NOMINAL_IMPEDANCE_FN[i]) 

          # example: with a=-. --> -50 - (-.5) * (-100 - (-50)) = -50 + . 5*(-50) - -75
      #print(action)
      #print('pos:',u_pos,'ffwd',u_force)

      action = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),u_force)

    elif self._motor_control_mode == "TORQUE":
      # action is some fraction of torque limits
      action = action * self._robot_config.TORQUE_LIMITS
    else:
      raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
    return action

  def _transform_action_to_motor_command(self, action):
    #if self._leg_model_enabled:
    # going to assume this is already the case - actions get clipped afterwards anyways
    for i, action_component in enumerate(action):
      if not (-self._action_bound - ACTION_EPS <= action_component <=
              self._action_bound + ACTION_EPS):
        action[i] = np.clip(action[i], -self._action_bound - ACTION_EPS,self._action_bound + ACTION_EPS)
        #raise ValueError("{}th action {} out of bounds.".format(i, action_component))
    """ CHANGED """
    #action = self._robot.ConvertFromLegModel(action)
    if self._motor_control_mode == "PD":
      action = self._robot.ScaleActionToPosRange(action)
    elif self._motor_control_mode == "IK":
      action = self._robot.ScaleActionToIKRange(action)
    # elif self._motor_control_mode in ["IMPEDANCE","IMPEDANCE_POS_ONLY", "IMPEDANCE_GAINS","IMPEDANCE_3D_FORCE","IMPEDANCE_FULL"]:
    #   action, _ = self._robot.ScaleActionToImpedanceControl(action)
    elif self._motor_control_mode in ["IMPEDANCE_POS_ONLY", "IMPEDANCE_POS_GAINS"]:
      action = self.ScaleActionToImpedancePosOnly(action)
    elif self._motor_control_mode in ["IMPEDANCE_POS_ONLY_IK"]:
      action = self.ScaleActionToImpedancePosOnly(action,useJointPD=True)
    elif self._motor_control_mode in ["IMPEDANCE_POS_ONLY_SPHERICAL"]:
      # action = self.ScaleActionToImpedanceControlSpherical(action)
      action = self.ScaleActionToImpedanceControlLegSpace(action)
    elif self._motor_control_mode in ["IMPEDANCE_POS_ONLY_SPHERICAL_IK"]:
      # action = self.ScaleActionToImpedanceControlSpherical(action,useJointPD=True)
      action = self.ScaleActionToImpedanceControlLegSpace(action,useJointPD=True)
    elif self._motor_control_mode in ["IMPEDANCE_POS_VEL_ONLY"]:
      action, _ = self._robot.ScaleActionToImpedanceControl(action, velFlag=True)
    elif self._motor_control_mode in ["IMPEDANCE_TROT"]:
      action = self._robot.ScaleActionDiagImpedanceControl(action)
    elif self._motor_control_mode == "TORQUE":
      action = self._robot.ScaleActionToTorqueRange(action)
    elif "PMTG" in self._motor_control_mode: # in ["PMTG", "PMTG_XY_ONLY", "PMTG_JOINT", "PMTG_XY_ONLY_JOINT"]:
      action = self._get_PMTG_action(action)
    elif self._motor_control_mode in ["BEZIER_GAIT", "BEZIER_GAIT_JOINT"]:
      action = self._get_Bezier_action(action)
    else:
      raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
    return action

  def _action_filter(self,action,mode='rl'):
    """ Smooth actions. """
    lam = self._action_filter_weight
    #lam = 0.3 #, 0.1, 0.5 (tried)
    #return lam*action + (1-lam)*self._last_action
    return lam*action + (1-lam)*self._last_action_rl

  def ScaleActionToImpedancePosOnly(self,actions,useJointPD=False, useForceIfInContact=False):
    """Over writing quadruped.py TEMPORARILY """
    u = np.clip(actions,-1,1)
    # scale des xyz 
    low_lim_ik = self._robot_config.IK_POS_LOW_LEG
    upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
    u_pos = self._robot._scale_helper(u[:12], low_lim_ik, upp_lim_ik) 
    # feedforward forces are 0
    if self._motor_control_mode == "IMPEDANCE_POS_GAINS":
      # upp_gains = np.array([2000, 100])
      # low_gains = np.array([500, 0])\
      upp_gains = np.array([3000, 100])
      low_gains = np.array([300, 0])
      gains =  self._robot._scale_helper(u[12:14], low_gains, upp_gains) 
      kpCartesian = np.diag([gains[0]]*3)
      kdCartesian = np.diag([gains[1]]*3)
    else:
      kpCartesian = self._robot_config.kpCartesian
      kdCartesian = self._robot_config.kdCartesian
    kps = self._robot_config.MOTOR_KP[0:3]
    kds = self._robot_config.MOTOR_KD[0:3]

    # THIS WAS WORKING WELL (BEST RESULT SO FAR)
    kps = np.array([20]*3)
    kds = np.array([0.2]*3)

    # new tests (too much)
    # kps = np.array([30]*3)
    # kds = np.array([0.5]*3)

    # still stiff
    # kps = np.array([25]*3)
    # kds = np.array([0.3]*3)

    action = np.zeros(12)

    for i in range(4):
      u_force = np.zeros(3)
      # if useForceIfInContact:
      #   _,_,_,feetInContactBool = self._robot.GetContactInfo()

      #   for i,inContact in enumerate(feetInContactBool):
      #     if inContact == 1: # in contact
      #       u_force = np.array([0,0,self._robot_config.NOMINAL_IMPEDANCE_FN[i]])

      tau = self._robot._ComputeLegImpedanceControlActions(u_pos[3*i:3*i+3],
                                              np.zeros(3),
                                              kpCartesian,
                                              kdCartesian,
                                              u_force,
                                              i)

      action[3*i:3*i+3] = tau

      if useJointPD:
        # test with adding joint PD as well 
        # current angles
        q = self._robot.GetMotorAngles()[3*i:3*i+3]
        dq = self._robot.GetMotorVelocities()[3*i:3*i+3]
        # corresponding desired joint angles:
        q_des = self._robot._ComputeIKSpotMicro(i, u_pos[3*i:3*i+3] ) 
        dq_des = np.zeros(3)
        tau_joint = kps*(q_des-q) + kds*(dq_des-dq)

        action[3*i:3*i+3] += tau_joint

    return action

  def ScaleActionToImpedanceControlSpherical(self,actions,useJointPD=False):
    """Scale action from [-1,1] to spherical coordinates, then to foot xyz position 
    r in [0.15,0.35], theta in [-0.8,0.8], psi in [-0.8,0.8]
    UPDATE: psi should be in -pi to pi, theta in 0, 0.8
    """
    # print('='*40)
    # print(actions)
    # low_lim = np.array([0.15, -0.8, -0.8]*4)
    # upp_lim = np.array([0.35,  0.8,  0.8]*4)
    # # low_lim = np.array([0.15, -0.8,       0 ]*4)
    # upp_lim = np.array([0.35, 0.8, 2*np.pi]*4)
    low_lim = np.array([0.15, 0,   -np.pi]*4)
    upp_lim = np.array([0.35, 0.8, np.pi]*4)
    # lower range a bit for theta 
    upp_lim = np.array([0.35, 0.6, np.pi]*4)
    upp_lim = np.array([0.33, 0.4, np.pi]*4)

    u = np.clip(actions,-1,1)
    u = self._robot._scale_helper(u, low_lim, upp_lim) 

    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian
    kps = self._robot_config.MOTOR_KP[0:3]
    kds = self._robot_config.MOTOR_KD[0:3]

    # kpCartesian = np.diag([500,500,500])
    # kdCartesian = np.diag([10,10,10])
    # kps = np.array([50]*3)
    # kds = np.array([0.5]*3)

    kps = np.array([25]*3)
    kds = np.array([0.3]*3)
    # kps = np.array([10]*3)
    # kds = np.array([0.1]*3)
    kps = np.array([20]*3)
    kds = np.array([0.2]*3)


    # kpCartesian = np.diag([2000,2000,2000])
    # kdCartesian = np.diag([50,50,50])
    # kps = np.array([20]*3)
    # kds = np.array([1]*3)


    # # 1000, 15, 50, 1
    # kpCartesian = np.diag([1000]*3)
    # kdCartesian = np.diag([15]*3)
    # kps = np.array([50]*3)
    # kds = np.array([1]*3)


    action = np.zeros(12)

    for i in range(4):
      spherical_coords = self._robot.spherical2cartesian(u[3*i],u[3*i+1],u[3*i+2])
      if i==0 or i==2:
        hip = np.array([0, -self._robot_config.HIP_LINK_LENGTH, 0])
      else:
        hip = np.array([0,  self._robot_config.HIP_LINK_LENGTH, 0])

      xyz = np.array([1,1,-1])*spherical_coords + hip
      # print(i, xyz, u[3*i],u[3*i+1],u[3*i+2])

      tau = self._robot._ComputeLegImpedanceControlActions(xyz,
                                              np.zeros(3),
                                              kpCartesian,
                                              kdCartesian,
                                              np.zeros(3),
                                              i)

      action[3*i:3*i+3] = tau

      if useJointPD:
        # test with adding joint PD as well 
        # current angles
        q = self._robot.GetMotorAngles()[3*i:3*i+3]
        dq = self._robot.GetMotorVelocities()[3*i:3*i+3]
        # corresponding desired joint angles:
        q_des = self._robot._ComputeIKSpotMicro(i, xyz ) 
        dq_des = np.zeros(3)
        tau_joint = kps*(q_des-q) + kds*(dq_des-dq)

        action[3*i:3*i+3] += tau_joint

    return action

  def ScaleActionToImpedanceControlLegSpace(self,actions,useJointPD=False):
    """Scale action from [-1,1] to leg space coordinates, then to foot xyz position 
    hip can vary within limits, 
    front/back can be -pi/3, pi/3
    radius in 0.15,0.35

    RUN again decreasing hip range (too large)
    """

    low_lim = np.array([-0.6, -np.pi/3, 0.15]*4)
    upp_lim = np.array([ 0.6,  np.pi/3, 0.33]*4)

    # probably plenty 
    low_lim = np.array([-0.4, -np.pi/3, 0.15]*4)
    upp_lim = np.array([ 0.4,  np.pi/3, 0.33]*4)

    u = np.clip(actions,-1,1)
    u = self._robot._scale_helper(u, low_lim, upp_lim) 

    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian
    kps = self._robot_config.MOTOR_KP[0:3]
    kds = self._robot_config.MOTOR_KD[0:3]

    # kps = np.array([20]*3)
    # kds = np.array([0.2]*3)
    kps = np.array([25]*3)
    kds = np.array([0.3]*3)

    # kps = np.array([10]*3)
    # kds = np.array([0.1]*3)

    # kps = np.array([50]*3)
    # kds = np.array([0.5]*3)

    action = np.zeros(12)

    for i in range(4):
      #spherical_coords = self._robot.spherical2cartesian(u[3*i],u[3*i+1],u[3*i+2])
      # get joint angles
      legi = np.zeros(3)
      legi[0] = u[3*i]
      # get thigh and calf angles 
      theta = u[3*i+1]
      radius = u[3*i+2]
      z = -radius*np.cos(theta)
      x =  radius*np.sin(theta)
      # 2D IK just to get thigh and calf angles
      xz = self._robot._ComputeIKSpotMicro(i, np.array([x,0,z]) )
      legi[1:] = xz[1:]
      # now that we have leg angles, compute xyz position
      _, xyz = self._robot._ComputeJacobianAndPositionUSC(i,specific_delta_q=legi)

      # if i==0 or i==2:
      #   hip = np.array([0, -self._robot_config.HIP_LINK_LENGTH, 0])
      # else:
      #   hip = np.array([0,  self._robot_config.HIP_LINK_LENGTH, 0])

      # xyz = np.array([1,1,-1])*spherical_coords + hip
      # print(i, xyz, u[3*i],u[3*i+1],u[3*i+2])

      tau = self._robot._ComputeLegImpedanceControlActions(xyz,
                                              np.zeros(3),
                                              kpCartesian,
                                              kdCartesian,
                                              np.zeros(3),
                                              i)

      action[3*i:3*i+3] = tau

      if useJointPD:
        # test with adding joint PD as well 
        # current angles
        q = self._robot.GetMotorAngles()[3*i:3*i+3]
        dq = self._robot.GetMotorVelocities()[3*i:3*i+3]
        # corresponding desired joint angles:
        # don't recompute
        # q_des = self._robot._ComputeIKSpotMicro(i, xyz ) 
        q_des = legi
        dq_des = np.zeros(3)
        tau_joint = kps*(q_des-q) + kds*(dq_des-dq)

        action[3*i:3*i+3] += tau_joint

    return action

  def _get_PMTG_action(self,action):
    """ Calculate actions according to PMTG. 
    First 4 values are frequencies - (what are good upper/lower limits to modulate?)
    Next 8 are xy offsets for feet
    """
    USE_MAX_DELTA = False
    DEBUG_LINE = False
    MAX_DELTA = 0.01

    if self._motor_control_mode in ["PMTG_XY_ONLY", "PMTG_XY_ONLY_JOINT"]:
      # just xys, move to xyz
      new_a = np.zeros(12)
      for i in range(4):
        new_a[3*i:3*i+2] = action[2*i:2*i+2] 
      action = np.concatenate(( np.zeros(4), new_a))
    elif self._motor_control_mode in ["PMTG", "PMTG_JOINT"]:
      # first 4 are frequencies, had only XY
      temp_a = action[4:]
      new_a = np.zeros(12)
      for i in range(4):
        new_a[3*i:3*i+2] = temp_a[2*i:2*i+2] 
      action = np.concatenate(( action[:4], new_a))
    elif self._motor_control_mode in ["PMTG_XYZ_ONLY", "PMTG_XYZ_ONLY_JOINT"]:
      # no frequencies
      action = np.concatenate(( np.zeros(4), action))
    # elif self._motor_control_mode in 

    nominal_foot_xyz = self._robot_config.NOMINAL_IMPEDANCE_POS
    nominal_joint_offsets= self._robot_config.JOINT_OFFSETS

    # these are upper/lower limits on adding frequency offset (so can slow down or speed up base freq)
    upp_fi =  .5 #1#/0.2
    low_fi = 0.1 #-1#/0.2
    # upp_fi =  1 # 1#/0.2
    # low_fi = -1 #-1#/0.2
    # upp_fi =  .4 # 1#/0.2
    # low_fi = -.4 #-1#/0.2
    #fi = self._robot._scale_helper( action[:4], low_fi, upp_fi) 
    fi = upp_fi * (action[:4] + 1)/2

    # fi = np.zeros(4)
    # should be based on period
    #ftg_T = self._ftg.getNominalPeriod

    foot_dh = self._ftg.setPhasesAndGetDeltaHeights(self.get_sim_time(), fi=fi)

    upp_xyz = np.array([0.1, 0.05, 0.05]*4)
    xyz = action[4:] * upp_xyz

    # determine full u_pos
    # u_pos = np.zeros(12)
    # for i in range(4):
    #   # print('nominal foot xyz', nominal_foot_xyz[3*i:3*i+2], 'delta', xy[2*i:2*i+2], 'foot dh', foot_dh[i])
    #   # u_pos[3*i:3*i+2] = nominal_foot_xyz[3*i:3*i+2] + xy[2*i:2*i+2]
    #   u_pos[3*i:3*i+3] = nominal_foot_xyz[3*i:3*i+3] + xyz[3*i:3*i+3]
    #   # u_pos[3*i+2] = nominal_foot_xyz[3*i+2] + foot_dh[i]
    #   ################################################ CHANGED
    #   u_pos[3*i+2] += foot_dh[i]
    #   print('u_pos', u_pos[3*i:3*i+2], 'u_pos', u_pos[3*i+2])
    #   print(xyz)


    u_pos = np.zeros(12)
    for i in range(4):
      if i==0 or i==2:
        FR_hip = np.array([0, -self._robot_config.HIP_LINK_LENGTH, 0])
      else:
        FR_hip = np.array([0,  self._robot_config.HIP_LINK_LENGTH, 0])
      u_pos[3*i:3*i+3] = FR_hip + xyz[3*i:3*i+3]
      nom_z = 0.25 #0.3
      u_pos[3*i+2] += -(nom_z - foot_dh[i])


    # clip basd on current foot positions - avoid too large a change
    if USE_MAX_DELTA:
      for i in range(4):

        _, curr_ee_pos = self._robot._ComputeJacobianAndPositionUSC(i)
        # u_pos[3*i:3*i+2] = np.clip(u_pos[3*i:3*i+2], curr_ee_pos[0:2]-MAX_DELTA, curr_ee_pos[0:2]+MAX_DELTA)
        u_pos[3*i:3*i+3] = np.clip(u_pos[3*i:3*i+3], curr_ee_pos-MAX_DELTA, curr_ee_pos+MAX_DELTA)
        # print(i, 'curr', curr_ee_pos, 'clipped', u_pos[3*i:3*i+2] )
        # print('des foot pos',i, u_pos[3*i:3*i+3])

    # time.sleep(0.01)
    # print('des foot pos',u_pos)
    if DEBUG_LINE:
      for i in range(4):
        _, curr_ee_pos = self._robot._ComputeJacobianAndPositionUSC(i)
        #u_pos[3*i:3*i+2] = np.clip(u_pos[3*i:3*i+2], curr_ee_pos[0:2]-MAX_DELTA, curr_ee_pos[0:2]+MAX_DELTA)
        base_hip_offset = np.array(self._robot.GetBasePosition()) + self._robot_config.HIP_POSITIONS[i]
        self.addLine(base_hip_offset+curr_ee_pos, base_hip_offset+u_pos[3*i:3*i+3])

    if "JOINT" in self._motor_control_mode:
      # calculate IK
      joint_pos = np.zeros(12)
      # since IK are in body frame, add hip offset
      hip_pos = self._robot_config.HIP_POSITIONS 
      for i in range(4):
        legi_joint_pos = self._robot._ComputeIK(i, u_pos[3*i:3*i+3] + hip_pos[i])
        # legi_joint_pos = self._robot._ComputeInverseKinematicsUSC(i,u_pos[3*i:3*i+3])
        # print('joint pos leg',i,legi_joint_pos)
        joint_pos[3*i:3*i+3] = legi_joint_pos + nominal_joint_offsets[3*i:3*i+3]

      # self._robot.SetJointStates(np.concatenate((joint_pos,np.zeros(12))))
      # time.sleep(0.1)
      return joint_pos
    else:
      # do impedance control
      u_force = self._robot_config.NOMINAL_IMPEDANCE_FN
      u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])
      # u_force
      u_force = np.zeros(12)
      a = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),u_force)

      useJointPD = True
      if useJointPD:
        # kps = np.array([20]*3)
        # kds = np.array([0.2]*3)
        kps = np.array([50]*3)
        kds = np.array([1]*3)
        for i in range(4):    
          # test with adding joint PD as well 
          # current angles
          q = self._robot.GetMotorAngles()[3*i:3*i+3]
          dq = self._robot.GetMotorVelocities()[3*i:3*i+3]
          # corresponding desired joint angles:
          q_des = self._robot._ComputeIKSpotMicro(i, u_pos[3*i:3*i+3] ) 
          dq_des = np.zeros(3)
          tau_joint = kps*(q_des-q) + kds*(dq_des-dq)

          a[3*i:3*i+3] += tau_joint
      # print('torques:', a)
      return a

  def _get_Bezier_action(self,action):
    """ Calculate actions according to Bezier Gait.
    Add to OpenLoop values? 

    """
    # action = 0*np.ones(12)
    # import pdb
    # pdb.set_trace()
    print('\n')
    print('='*60)
    USE_MAX_DELTA = False
    DEBUG_LINE = False
    MAX_DELTA = 0.01
    # print('action', action)
    # if len(action) == 8:
    #   # pos only
    #   action = np.concatenate(( np.zeros(4), action))

    nominal_foot_xyz = self._robot_config.NOMINAL_IMPEDANCE_POS
    nominal_joint_offsets= self._robot_config.JOINT_OFFSETS

    # # these are upper/lower limits on adding frequency offset (so can slow down or speed up base freq)
    # upp_fi =  .5 #1#/0.2
    # low_fi = 0.1 #-1#/0.2
    # fi = upp_fi * (action[:4] + 1)/2


    # foot_dh = self._ftg.setPhasesAndGetDeltaHeights(self.get_sim_time(), fi=fi)
    _,_,_,feetInContactBool = self._robot.GetContactInfo()
    # feetInContactBool = [1,1,1,1]
    L = 0.01

    LateralFraction = 0#1.57
    YawRate = 0#0.6
    vel = 0.01
    
    clearance_height = 0.025
    penetration_depth = 0.01 #0.01
    # test ramp up
    # L = self.get_sim_time() / 200
    vel = self.get_sim_time() / 5
    YawRate = 0 #self.get_sim_time() / 2

    # Phase Lag Per Leg: FL, FR, BL, BR
    # foot_names = ["FL", "FR", "BL", "BR"]
    foot_names = ["FR", "FL", "BR", "BL"]
    keys = [1, 0, 3, 2]
    T_bf = {}
    foot_pos = {}
    for i, foot in enumerate(foot_names):
      T_bf[foot] = self._robot.GetLocalEEPos(i)
      # _, T_bf[foot] = self._robot._ComputeJacobianAndPositionUSC(i

      # from DEfault, actual 
      T_bf[foot] = self._robot_config.DEFAULT_LOCAL_EE_POS[i]


    print('actual foot pos',T_bf)
    ## Bezier
    foot_pos = self._bzg.GenerateTrajectory(
                           L,
                           LateralFraction,
                           YawRate,
                           vel,
                           T_bf,
                           #T_bf_curr,
                           clearance_height=clearance_height,#0.02, # 0.06
                           penetration_depth=penetration_depth,#0.01,
                           contacts=feetInContactBool,
                           dt=None)

    ### test bezier swings
    swing_duration = .2
    phase = (self.get_sim_time() % swing_duration) / swing_duration

    # test BezierSwing
    # for i, foot in enumerate(foot_names):
    #   dx, dy, dz = self._bzg.BezierSwing(phase, L, LateralFraction, clearance_height=0.05)
    #   foot_pos[foot] = nominal_foot_xyz[3*i:3*i+3] +  np.array([dx,dy,dz])

    # test SwingStep
    # Notes: clearance_height is doubled due to yaw contribution 
    # for i, foot in enumerate(foot_names):
    #   dx, dy, dz = self._bzg.SwingStep(phase, L, LateralFraction, YawRate, clearance_height, T_bf[foot], foot, i)
    #   foot_pos[foot] = nominal_foot_xyz[3*i:3*i+3] +  np.array([dx,dy,dz])

    # test GetFootStep
    #   Notes: 
    # increment

    # Tstance = 0.2
    # self._bzg.Increment(self._bzg.dt, Tstance + self._bzg.Tswing)
    # # for i, foot in zip(keys,foot_names) : #enumerate(foot_names):
    # for i, foot in enumerate(foot_names):
    #   foot_pos[foot] = self._bzg.GetFootStep( L, LateralFraction, YawRate, clearance_height,
    #                 penetration_depth, Tstance, T_bf[foot], i, foot)
    #   foot_pos[foot] += nominal_foot_xyz[3*i:3*i+3]

    # time.sleep(0.01)
    print('time', self.get_sim_time())
    print('des foot pos',foot_pos)
    if DEBUG_LINE:
      for i in range(4):
        _, curr_ee_pos = self._robot._ComputeJacobianAndPositionUSC(i)
        #u_pos[3*i:3*i+2] = np.clip(u_pos[3*i:3*i+2], curr_ee_pos[0:2]-MAX_DELTA, curr_ee_pos[0:2]+MAX_DELTA)
        base_hip_offset = np.array(self._robot.GetBasePosition()) + self._robot_config.HIP_POSITIONS[i]
        # self.addLine(base_hip_offset+curr_ee_pos, base_hip_offset+u_pos[3*i:3*i+3])
        # self.addLine(base_hip_offset+curr_ee_pos, base_hip_offset+foot_pos[foot_names[i]])
        self.addLine(base_hip_offset+curr_ee_pos, np.array(self._robot.GetBasePosition()) + foot_pos[foot_names[i]])


    hip_pos = self._robot_config.HIP_POSITIONS  
    if "JOINT" in self._motor_control_mode:
      joint_pos = np.zeros(12)
      hip_pos = self._robot_config.HIP_POSITIONS
      for i, foot_name in enumerate(foot_names):
      # for i, foot_name in zip(keys,foot_names) :
        legi_joint_pos = self._robot._ComputeIK(i, foot_pos[foot_name])
        # legi_joint_pos = self._robot._ComputeIK(i, foot_pos[foot_name] + np.array([0,0,-0.3])) # hip_pos[i][2]
        # legi_joint_pos = self._robot._ComputeIK(i, foot_pos[foot_name] + hip_pos[i])
        # legi_joint_pos = self._robot._ComputeIK(i, foot_pos[foot_name] + hip_pos[i] + np.array([0,0,-0.3]))
        # test spotMicro IK
        # legi_joint_pos = self._robot._ComputeIKSpotMicro(i, foot_pos[foot_name])
        legi_joint_pos = self._robot._ComputeIKSpotMicro(i, foot_pos[foot_name] - hip_pos[i])
        joint_pos[3*i:3*i+3] = legi_joint_pos + nominal_joint_offsets[3*i:3*i+3]

      # self._robot.SetJointStates(np.concatenate((joint_pos,np.zeros(12))))
      return joint_pos

    else:
      # impedance control instead
      #   # do impedance contro
      u_pos = np.zeros(12)
      for i, foot_name in enumerate(foot_names):
        # u_pos[3*i:3*i+3] = foot_pos[foot_name]
        u_pos[3*i:3*i+3] = foot_pos[foot_name] - hip_pos[i]

      u_force = self._robot_config.NOMINAL_IMPEDANCE_FN
      u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])
      # u_force
      u_force = np.zeros(12)
      a = self._robot._ComputeFullImpedanceControlActions(u_pos,np.zeros(12),u_force)
      # print('torques:', a)
      return a



  def step_full_traj(self,action):
    """ Step for full traj in action space. """

    # reshape action into number of offsets to use
    #reshaped_action = np.reshape(action,(12,-1))

    reshaped_action = np.reshape(action,(self._action_dim_orig_action_space, self._output_mode_num))
    #print('reshaped_action', reshaped_action)

    for i in range(self._output_mode_num):

      for _ in range(self._action_repeat):
        proc_action = self._transform_action_to_motor_command(reshaped_action[:,i])

        self._robot.ApplyAction(proc_action)
        self._pybullet_client.stepSimulation()
        self._sim_step_counter += 1
        self._last_action = self._robot._applied_motor_torque
      
        if self._is_render:
          time.sleep(0.001)
          self._render_step_helper()

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination() or self.get_sim_time() > self._MAX_EP_LEN
    return np.array(self._noisy_observation()), reward, done, {'base_pos': self._robot.GetBasePosition()} # self.getPybData() #


  def step(self, action):
    """ Step forward the simulation, given the action. """
    # don't need this here
    # if self._is_render:
    #   self._render_step_helper()

    if self._output_mode == "N_TRAJ":
      return self.step_full_traj(action)

    #print('action', action, 'time', self.get_sim_time())
    # rgbImg, depthImg = self.get_robot_cam_view()
    # rgbImg = np.array(rgbImg)
    # depthImg = np.array(depthImg)
    # print(rgbImg.shape,depthImg.shape)

    # self._orig_action = action.copy()
    # self._prev_rl_foot_pos = np.zeros(12)
    # ee_pos_leg_frames = np.zeros(12)
    # for i in range(4):
    #   _, ee_pos_leg_frame = self._robot._ComputeJacobianAndPositionUSC(i)
    #   self._prev_rl_foot_pos[i*3:i*3+3] = ee_pos_leg_frame 
    #   #self._robot.GetLocalEEPos(i) # this is w.r.t. body
    #   #ee_pos_leg_frames[i*3:i*3+3] = ee_pos_leg_frame
    # print('get local ee pos', self._prev_rl_foot_pos)
    # #print('frm Jacobian pos', ee_pos_leg_frames)
    # if self._enable_action_filter:
    #   # make sure foot pos don't chang too fast
    #   if self._motor_control_mode != "IMPEDANCE_POS_ONLY":
    #     raise ValueError('need to implement action filtering for motor control mode ', self._motor_control_mode)

    curr_act = action.copy()
    if self._enable_action_filter:
      # print('last action', self._last_action_rl)
      curr_act = self._action_filter(curr_act)
      #print('action',curr_act)
    #action = np.zeros(12)

    #curr_act = np.zeros(12)
    # apply random perturbation? 
    perturbation = np.zeros(3)
    # if self._terrain_noise:
    #   if np.random.random() < self._external_force_probability: 
    #     force_vec = 2*(np.random.random(3) - 0.5)
    #     force_mag = 1000
    #     perturbation = force_mag*force_vec
    #     #print('applying force', force_mag*force_vec, 'at base')
    #     self._robot.ApplyExternalForce(force_mag*force_vec)

    #     #self._pybullet_client.getBasePositionAndOrientation(self.quadruped)
    self._dt_motor_torques = []
    self._dt_motor_velocities = []
    
    for _ in range(self._action_repeat):
      # only scale action if it's a gym interface
      self._robot.ApplyExternalForce(perturbation)
      #print('-----------------curr_act in loop', curr_act)
      if self._GAS:
        if self._scale_GAS_action:
          proc_action = self._transform_action_GAS(curr_act)
        else:
          proc_action = self._transform_action_GAS_2(curr_act)
      elif self._isRLGymInterface: # this is doing the same as _leg_model_enabled
        proc_action = self._transform_action_to_motor_command(curr_act)
      else:
        proc_action = curr_act #action
      self._robot.ApplyAction(proc_action)
      self._pybullet_client.stepSimulation()
      self._sim_step_counter += 1
      self._dt_motor_torques.append(self._robot.GetMotorTorques())
      self._dt_motor_velocities.append(self._robot.GetMotorVelocities())
      if self._is_render:
        self._render_step_helper()

    # this will never happen, remove?
    # if any(abs(self._robot.GetMotorVelocities()) > self._robot_config.VELOCITY_LIMITS) \
    #   or any(abs(self._robot.GetMotorTorques()) > self._robot_config.TORQUE_LIMITS):
    #   print('MOTOR VIOLATION')
    #   print('motor velocities:', self._robot.GetMotorVelocities())
    #   print('motor torques:   ', self._robot.GetMotorTorques()  )


    self._last_action_rl = curr_act #action
    self._robot._last_action_rl = curr_act

    if DEBUG_FLAG:
      self._print_robot_states()
      time.sleep(1)

    self._env_step_counter += 1
    done = False
    reward = self._reward()
    # done = self._termination() or self.get_sim_time() > self._MAX_EP_LEN
    # done = done or self.get_sim_time() > self._MAX_EP_LEN
    if self._termination():
      done = True
      reward = -10 # tried -100 (?)
    if self.get_sim_time() > self._MAX_EP_LEN:
      done = True
      # reward = -10
    if self._terminate:
      done = True
      # reward = 10
    #print(np.array(self._noisy_observation()), reward, done)
    return np.array(self._noisy_observation()), reward, done, {'base_pos': self._robot.GetBasePosition()} # self.getPybData() #

  def _print_robot_states(self):
    """Print relevant robot states, for debugging """
    # print('*'*80)
    # _,_,_,feetInContactBool = self._robot.GetContactInfo()
    # for c in self._pybullet_client.getContactPoints():
    #   print( c )
    # print('feetInContactBool', feetInContactBool)
    # print('feet indices', self._robot._foot_link_ids )
    print('*'*80)
    print(self._noisy_observation())
    
    _,_,_,feetInContactBool = self._robot.GetContactInfo()
    print('contacts', feetInContactBool)
    #R = self.GetBaseOrientationMatrix()
    print('base linvel',self._robot.GetBaseLinearVelocity())
    print('base angvel local',self._robot.GetTrueBaseRollPitchYawRate())
    print('base angvel world',self._robot.GetBaseAngularVelocity())
    print('joint angles',self._robot.GetMotorAngles())

    print('='*20)
    # add foot pos/ foot vel 

    for i in range(4):
      print('-'*10, i)
      foot_pos = self._robot.GetLocalEEPos(i)
      # Jacobian and current foot position in hip frame
      J, ee_pos_legFrame = self._robot._ComputeJacobianAndPositionUSC(i)
      # foot velocity in leg frame
      foot_linvel = J @ self._robot.GetMotorVelocities()[i*3:i*3+3]
      print('foot pos pybullet', foot_pos)
      print('foot vel', foot_linvel)

    print(self._robot.GetFootContactForces())
  ######################################################################################
  # Termination and reward
  ######################################################################################
  def is_fallen(self,dot_prod_min=0.85):
    """Decide whether the quadruped has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the quadruped is considered fallen.

    Returns:
      Boolean value that indicates whether the quadruped has fallen.
    """
    orientation = self._robot.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self._robot.GetBasePosition()
    # return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < 0.13) # from minitaur
    #print(np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)))
    #print(np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < self._robot_config.IS_FALLEN_HEIGHT)
    dot_prod_min = 0.5
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < dot_prod_min or pos[2] < self._robot_config.IS_FALLEN_HEIGHT)

    """
    Check these limits.. though the invalid contact check may be enough anyways 

    """
    # return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.7 or pos[2] < self._robot_config.IS_FALLEN_HEIGHT)

  def _termination(self):
    position = self._robot.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    _, numInvalidContacts, _, _ = self._robot.GetContactInfo()
    # return self.is_fallen() or distance > self._distance_limit #or numInvalidContacts > 0
    return self.is_fallen() or distance > self._distance_limit or self._robot.ThighsInContact()

  def _reward_orig(self):
    """ Reward that worked originally for locomotion task. 
    i.e. how far the quadruped walks (x velocity) while penalizing energy
    """
    current_base_position = self._robot.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    self._last_base_position = current_base_position

    """ TODO: energy reward should be scaled by action_repeat as well(?) """

    energy_reward = np.abs(
        np.dot(self._robot.GetMotorTorques(),
               self._robot.GetMotorVelocities())) * self._time_step
    energy_reward = 0 
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

    # saturate max forward reward
    # fwd_reward_max = 0.05
    # if forward_reward > fwd_reward_max:
    #   forward_reward = fwd_reward_max
    if MAX_FWD_VELOCITY < np.inf:
      # calculate what max distance can be over last time interval
      max_dist = MAX_FWD_VELOCITY * (self._time_step * self._action_repeat)
      forward_reward = min( forward_reward, max_dist)

    reward = (self._distance_weight * forward_reward - self._energy_weight * energy_reward +
              self._drift_weight * drift_reward + self._shake_weight * shake_reward)
    self._objectives.append([forward_reward, energy_reward, drift_reward, shake_reward])
    return reward

  def _reward_fwd_locomotion(self):
    """ Add survival bonus to original reward (?) """
    survival_reward = 0.01
    orn_weight = 1e-2
    orn_reward = - orn_weight * np.linalg.norm(self._robot.GetBaseOrientation() - np.array([0,0,0,1]))
    y_reward = -5e-3 * abs(self._robot.GetBasePosition()[1])
    # y_reward = -1e-2 * abs(self._robot.GetBaseLinearVelocity()[1])
    height_reward = 0#- 1e-3 * abs(self._robot.GetBasePosition()[2] - 0.3) 
    return self._reward_orig() + survival_reward \
                              + orn_reward \
                              + y_reward \
                              + height_reward #.01 # some other factor could be better?

  def _reward_goal(self, xgoal=20, ygoal=0):
    """Get to goal as fast as possible """
    current_base_position = self._robot.GetBasePosition()
    # forward_reward = current_base_position[0] - self._last_base_position[0]
    # drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    # shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    # progress toward goal
    # fwd_reward = -np.sqrt(  (   current_base_position[0] - xgoal)**2  + (   current_base_position[1] - ygoal)**2 ) \
    #              +np.sqrt(  (self._last_base_position[0] - xgoal)**2  + (self._last_base_position[1] - ygoal)**2 )
    fwd_reward = -np.sqrt(  (   current_base_position[0] - xgoal)**2  ) \
                 +np.sqrt(  (self._last_base_position[0] - xgoal)**2  )


    energy_reward = 0 
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step
    orn_weight = .001
    time_penalty = 0#0.001
    reward = self._distance_weight * fwd_reward - self._energy_weight * energy_reward \
          - orn_weight * np.linalg.norm(self._robot.GetBaseOrientation() - np.array([0,0,0,1])) \
          - time_penalty

    self._last_base_position = current_base_position
    if abs(current_base_position[0] - xgoal) < 0.5:# and abs(current_base_position[1] - ygoal) < 0.5:
      self._terminate = True
    return reward




  def _reward_fwd_set_velocity(self,des_vel=1):
    """Learn locomotion at particular velocity """
    base_vel = self._robot.GetBaseLinearVelocity()
    shake_reward = -.2*abs(base_vel[2])
    drift_reward = -.2*abs(base_vel[1])
    energy_reward = np.abs(
        np.dot(self._robot.GetMotorTorques(),
               self._robot.GetMotorVelocities())) * self._time_step

    energy_reward = 0 
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

    des_vel = self._get_des_vel()
    #return des_vel - abs(base_vel[0]-des_vel) - self._energy_weight * energy_reward
    #return DES_VEL_HIGH - abs(base_vel[0] - self._des_velocity) - self._energy_weight * energy_reward
    #return 1 - abs(base_vel[0] - self._des_velocity) - self._energy_weight * energy_reward

    # from PMTG paper
    # return  DES_VEL_HIGH * np.exp( - (base_vel[0] - self._des_velocity)**2 / (2*DES_VEL_HIGH**2) )
    vel_reward = .1*(1 - abs(base_vel[0] - des_vel))
    # if vel_reward < 0:
    #   vel_reward = 0
    # orn_weight = 1e-3
    orn_weight = 0.1 #1e-2
    orn_reward = - orn_weight * np.linalg.norm(self._robot.GetBaseOrientation() - np.array([0,0,0,1]))
    y_reward = -5e-2 * abs(self._robot.GetBasePosition()[1])
    # return .1 * np.exp( - (base_vel[0] - des_vel)**2  ) - self._energy_weight * energy_reward \
    #       + orn_reward \
    #       + shake_reward + drift_reward
    return vel_reward + orn_reward - self._energy_weight * energy_reward + y_reward


  def _get_des_vel(self, ramp_up=True):
    """Get desired velocity, depends on time """
    if not ramp_up:
      return self._des_velocity
    des_vel = 0
    t = self.get_sim_time()
    # time_arr = [0.1, 1.5, self._MAX_EP_LEN - 1.5, self._MAX_EP_LEN - 0.3]
    # ramp up only 
    time_arr = [0.1, 2.5, self._MAX_EP_LEN * 2]
    if t < time_arr[0]:
      # stay standing 
      des_vel = 0
    elif t < time_arr[1]:
      # ramp up to self._des_velocity
      des_vel = (t - time_arr[0]) / (time_arr[1] - time_arr[0]) * self._des_velocity
    elif t < time_arr[2]:
      # should be at self._des_velocity
      des_vel = self._des_velocity
    elif t < time_arr[3]:
      # ramp down to 0
      des_vel = ( 1 - (t - time_arr[2]) / (time_arr[3] - time_arr[2]) ) * self._des_velocity
    else:
      # should be at 0 until end
      des_vel = 0

    return des_vel

  def _reward_stand(self, target_height=0.3):
    """ reward for standing at specific height"""
    curr_z = self._robot.GetBasePosition()[2]
    last_z = self._last_base_position[2]

    #drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    #shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    motor_vel_reward = np.sum(np.abs(self._robot.GetMotorVelocities()))
    motor_vel_reward = np.linalg.norm(self._robot.GetMotorVelocities())

    curr_orn = self._robot.GetBaseOrientationRollPitchYaw()
    orn_reward = curr_orn[0]**2 + curr_orn[1]**2
    energy_reward = np.abs(
        np.dot(self._robot.GetMotorTorques(),
               self._robot.GetMotorVelocities())) * self._time_step
    # if abs(target_height-curr_z) < .05:
    #   reward = 1
    # else:
    # note this was enough to get it to stand! but lots of shake..
    reward = target_height - abs(target_height-curr_z)

    #return reward - self._energy_weight * energy_reward - 0.1*orn_reward 
    return reward - self._energy_weight * energy_reward #- 0.05 * motor_vel_reward #- 0.1*orn_reward 

  def _reward(self):
    """ Get reward depending on task"""
    if self._TASK_ENV == "DEFAULT":
      return self._reward_orig()
    elif self._TASK_ENV == "STANDUP":
      return self._reward_stand()
    elif self._TASK_ENV == "FWD_LOCOMOTION":
      return self._reward_fwd_locomotion()
    elif self._TASK_ENV == "FWD_SET_VELOCITY":
      return self._reward_fwd_set_velocity()
    elif self._TASK_ENV == "GOAL":
      return self._reward_goal()
    else:
      raise ValueError("This task mode not implemented yet.")


  ######################################################################################
  # Observation
  ######################################################################################
  def get_objectives(self):
    return self._objectives

  def _get_observation(self):
    """Get observation, depending on obs space selected. """
    if self._observation_space_mode == "DEFAULT":
      self._observation = np.array(self._robot.GetObservation())
      self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
          self._robot.GetObservationUpperBound())
    elif self._observation_space_mode == "DEFAULT_CONTACTS":
      self._observation = np.array(self._robot.GetObservationContacts())
      # self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
      #     self._robot.GetObservationContactsUpperBound())
      # self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape))
    elif self._observation_space_mode == "EXTENDED":
      self._observation = np.array(self._robot.GetExtendedObservation())
      self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
          self._robot.GetExtendedObservationUpperBound())
    elif self._observation_space_mode == "EXTENDED2":
      self._observation = np.array(self._robot.GetExtendedObservation2())
      self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
          self._robot.GetExtendedObservation2UpperBound())
    elif self._observation_space_mode == "EXTENDED2VISION":
      self._observation = np.array(self._robot.GetExtendedObservation2Vision())
      self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
          self._robot.GetExtendedObservation2VisionUpperBound())
    else:
      raise ValueError("observation space not defined or not intended")
    #self._observation = self._robot.GetObservation()

    if self._TASK_ENV == "FWD_SET_VELOCITY":
      #self._observation = np.concatenate((self._observation, self._des_velocity ))
      self._observation = np.append(self._observation, self._get_des_vel() )

    if "PMTG" in self._motor_control_mode:
      self._observation = np.append(self._observation, self._ftg.getPhases() )

    if self._observation_history_length > 1:
      self._observation_history.append(self._observation)
      self._observation = np.concatenate(self._observation_history)

    if self._time_remaining_in_obs_space:
      self._observation = np.append(self._observation, self._MAX_EP_LEN - self.get_sim_time() )

    self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape))

    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += self._add_obs_noise
      # observation += (
      #     np.random.normal(scale=self._observation_noise_stdev, size=observation.shape) *
      #     self._robot.GetObservationUpperBound())

    if self._normalize_obs: # also check its initialized
      # TODO
      state = observation.copy()
      contacts = state[-4:]
      try:
        self.normalizer.observe(state)
        observation = self.normalizer.normalize(state)
      except:
        print('add normalizer')
        self.normalizer = Normalizer(len(observation))
      # undo contacts normalization
      #observation[-4:] = contacts

    return observation

  def get_sim_time(self):
    """ Get current simulation time from pybullet robot. """
    #return self._env_step_counter * self._action_repeat * self._time_step
    return self._sim_step_counter * self._time_step


  ######################################################################################
  # Render, record videos, and misc
  ######################################################################################
  def startRecordingVideo(self,name):
    self.videoLogID = self._pybullet_client.startStateLogging(
                            self._pybullet_client.STATE_LOGGING_VIDEO_MP4, 
                            name)

  def stopRecordingVideo(self):
    self._pybullet_client.stopStateLogging(self.videoLogID)

  def close(self):
    if self._is_record_video:
      self.stopRecordingVideo()
    self._pybullet_client.disconnect()

  def recordVideoHelper(self, extra_filename=None):
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

    if extra_filename is not None:
      output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") +extra_filename+ ".MP4"
    else:
      output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"
    logID = self.startRecordingVideo(output_video_filename)
    self.videoLogID = logID

  def set_env_randomizer(self, env_randomizer):
    self._env_randomizer = env_randomizer

  def configure(self, args):
    self._args = args

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _render_step_helper(self):
    """ Helper to configure the visualizer camera during step(). """
    # Sleep, otherwise the computation takes less time than real time,
    # which will make the visualization like a fast-forward video.
    time_spent = time.time() - self._last_frame_time
    self._last_frame_time = time.time()
    # time_to_sleep = self._action_repeat * self._time_step - time_spent
    time_to_sleep = self._time_step - time_spent
    if time_to_sleep > 0 and (time_to_sleep < self._time_step):
      time.sleep(time_to_sleep)
    base_pos = self._robot.GetBasePosition()
    camInfo = self._pybullet_client.getDebugVisualizerCamera()
    curTargetPos = camInfo[11]
    distance = camInfo[10]
    yaw = camInfo[8]
    pitch = camInfo[9]
    targetPos = [
        0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
        curTargetPos[2]
    ]
    self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)
    #self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, [0.3,0,.45]) # test for paper

  def _configure_visualizer(self):
    """ Remove all visualizer borders, and zoom in """
    # default rendering options
    self._render_width = 960
    self._render_height = 720
    # self._camera_dist = 1
    # self._camera_pitch = -30
    # self._camera_yaw = 0
    self._cam_dist = 1.0 # .75 #for paper
    self._cam_yaw = 0
    self._cam_pitch = -30 # -10 # for paper 
    # get rid of random visualizer things
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self._robot.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                   aspect=float(self._render_width) /
                                                                   self._render_height,
                                                                   nearVal=0.1,
                                                                   farVal=100.0)
    (_, _, px, _,
     _) = self._pybullet_client.getCameraImage(width=self._render_width,
                                               height=self._render_height,
                                               viewMatrix=view_matrix,
                                               projectionMatrix=proj_matrix,
                                               renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def addLine(self,lineFromXYZ,lineToXYZ,lifeTime=0,color=[1,0,0]):
    """ Add line between point A and B for duration lifeTime"""
    self._pybullet_client.addUserDebugLine(lineFromXYZ,
                                            lineToXYZ,
                                            lineColorRGB=color,
                                            lifeTime=lifeTime)
  ######################################################################################
  # Add randomness in terrain to experiment
  ######################################################################################
  def scale_rand(self,num_rand,low,high):
    """ scale number of rand numbers between low and high """
    return low + np.random.random(num_rand) * (high - low)

  def add_random_boxes(self, num_rand=100):
    """Add random boxes in front of the robot, should be in x [0.5, 50] and y [-5,5] 
    how many?
    how large?
    """
    #print('-'*80,'\nadding boxes\n','-'*80)
    # x location
    x_low = 0.5
    x_upp = 20
    # y location
    y_low = -3
    y_upp = 3
    # z, orig [0.01, 0.03]
    z_low = 0.005 # 
    z_upp = 0.025 # max height, was 0.025
    # z_upp = 0.04 # max height, was 0.025
    # block dimensions
    block_x_max = 1
    block_x_min = 0.1
    block_y_max = 1
    block_y_min = 0.1
    # block orientations
    # roll_low, roll_upp = -0.1, 0.1
    # pitch_low, pitch_upp = -0.1, 0.1
    roll_low, roll_upp = -0.01, 0.01
    pitch_low, pitch_upp = -0.01, 0.01 # was 0.001,
    yaw_low, yaw_upp = -np.pi, np.pi

    x = x_low + np.random.random(num_rand) * (x_upp - x_low)
    y = y_low + np.random.random(num_rand) * (y_upp - y_low)
    z = z_low + np.random.random(num_rand) * (z_upp - z_low)
    block_x = self.scale_rand(num_rand,block_x_min,block_x_max)
    block_y = self.scale_rand(num_rand,block_y_min,block_y_max)
    roll = self.scale_rand(num_rand,roll_low,roll_upp)
    pitch = self.scale_rand(num_rand,pitch_low,pitch_upp)
    yaw = self.scale_rand(num_rand,yaw_low,yaw_upp)
    # loop through
    for i in range(num_rand):
      sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
          halfExtents=[block_x[i]/2,block_y[i]/2,z[i]/2])
      orn = self._pybullet_client.getQuaternionFromEuler([roll[i],pitch[i],yaw[i]])
      block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [x[i],y[i],z[i]/2],baseOrientation=orn)
      # set friction coeff to 1
      self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add walls 
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.5,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)


  def add_random_boxes_horizontal(self, num_rand=30):
    """Wide, long so can't get around """
    x_low = 1
    x_upp = 20
    # y location
    y_low = -3
    y_upp = 3
    # z, orig [0.01, 0.03]
    z_low = 0.005 # 
    z_upp = 0.04 # max height, was 0.025
    # block dimensions
    block_x_max = 1
    block_x_min = 0.1
    y = 6

    x = x_low + np.random.random(num_rand) * (x_upp - x_low)
    z = z_low + np.random.random(num_rand) * (z_upp - z_low)
    block_x = self.scale_rand(num_rand,block_x_min,block_x_max)
    # loop through
    for i in range(num_rand):
      sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
          halfExtents=[block_x[i]/2,y/2,z[i]/2])
      orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
      block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [x[i],0,z[i]/2],baseOrientation=orn)
      # set friction coeff to 1
      self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add walls 
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.5,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)

  def make_rough_terrain(self,heightPerturbationRange=0.05,heightfieldSource='programmatic'):
    
    #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    row_size = 512
    
    if heightfieldSource=='programmatic':
      self.numHeightfieldRows = numHeightfieldRows = row_size #256
      self.numHeightfieldColumns = numHeightfieldColumns = row_size
      heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
      for j in range (int(numHeightfieldColumns/2)):
        for i in range (int(numHeightfieldRows/2) ):
          height = random.uniform(0,heightPerturbationRange)
          heightfieldData[2*i+2*j*numHeightfieldRows]=height
          heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
          heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
          heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
          
          
      terrainShape = self._pybullet_client.createCollisionShape(shapeType = self._pybullet_client.GEOM_HEIGHTFIELD, 
                                            meshScale=[.05,.05,1], 
                                            # meshScale=[.1,.1,1], 
                                            heightfieldTextureScaling=(numHeightfieldRows-1)/2, 
                                            heightfieldData=heightfieldData, 
                                            numHeightfieldRows=numHeightfieldRows, 
                                            numHeightfieldColumns=numHeightfieldColumns)
      self.plane = terrain  = self._pybullet_client.createMultiBody(0, terrainShape)
      self.terrainShape = terrainShape
      self._pybullet_client.resetBasePositionAndOrientation(terrain,[10,0,0], [0,0,0,1])

  def update_rough_terrain(self):
    numHeightfieldRows = self.numHeightfieldRows
    numHeightfieldColumns = self.numHeightfieldColumns
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0,heightPerturbationRange)#+math.sin(time.time())
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
    #GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
    #GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
    #flags = p.GEOM_CONCAVE_INTERNAL_EDGE
    flags = 0
    terrainShape2 = p.createCollisionShape(shapeType = self._pybullet_client.GEOM_HEIGHTFIELD, 
                                            flags = flags, meshScale=[.05,.05,1], 
                                            heightfieldTextureScaling=(numHeightfieldRows-1)/2, 
                                            heightfieldData=heightfieldData, 
                                            numHeightfieldRows=numHeightfieldRows, 
                                            numHeightfieldColumns=numHeightfieldColumns, 
                                            replaceHeightfieldIndex = self.terrainShape)
    

  def add_box(self):
    """ add box to env, temp test """
    box_x = 1
    box_y = 1
    box_z = 0.5
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,0.01])
    sth=0.25
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [box_x-0.2,0,box_z],baseOrientation=[0,0,0,1])

  def add_box_at(self,height,distance):
    """ add box to env, temp test """
    box_x = 1
    box_y = 1
    box_z = height
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])
    sth=0.25
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [distance+box_x/2-0.2,0,box_z/2],baseOrientation=[0,0,0,1])

  def add_box2(self):
    """ add box to env, temp test """
    box_x = 1
    box_y = 1
    box_z = 0.5
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])
    sth=0.25
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [box_x-0.15,0,box_z/2],baseOrientation=[0,0,0,1])

  def add_box_ff(self, z_height=0.025):
    """ add box under front feet, temp test """
    #box_x = 0.2
    box_x = 0.4
    box_y = 0.5
    box_z = 2*z_height #0.05
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])

    self.box_ff=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [box_x/2,0,0],baseOrientation=[0,0,0,1])

  def add_box_rr(self, z_height=0.025):
    """ add box under rear feet, temp test """
    #box_x = 0.2
    box_x = 0.4
    box_y = 0.5
    box_z = 2*z_height #0.05
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])

    # self.box_rr=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                         basePosition = [-0.25,0,0],baseOrientation=[0,0,0,1])
    self.box_rr=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [-box_x/2,0,0],baseOrientation=[0,0,0,1])

  def add_box_each_foot(self, z_height):
    """ add box under each foot, temp test """
    box_x = 0.2
    box_y = 0.25
    box_z = 2*z_height[0] #0.05

    # FR
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])

    self.box_FR=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [0.15,-box_y/2,0],baseOrientation=[0,0,0,1])

    # FL
    box_z = 2*z_height[1] #0.05
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])
    self.box_FL=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [0.15,box_y/2,0],baseOrientation=[0,0,0,1])

    # RR
    box_z = 2*z_height[2] #0.05
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])
    self.box_RR=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [-0.25,-box_y/2,0],baseOrientation=[0,0,0,1])

    # RL
    box_z = 2*z_height[3] #0.05
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
              halfExtents=[box_x/2,box_y/2,box_z/2])
    self.box_RL=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [-0.25,box_y/2,0],baseOrientation=[0,0,0,1])

  # if parse_version(gym.__version__) < parse_version('0.9.6'):
  #   _render = render
  #   _reset = reset
  #   _seed = seed
  #   _step = step

  def get_robot_cam_view(self):
    return self._robot.get_cam_view()

def main3leg():
  """ Test balancing on 3 legs """
  env = QuadrupedGymEnv(render=True, 
                        on_rack=False,
                        accurate_motor_model_enabled=True,
                        pd_control_enabled=False,
                        randomize_dynamics=False,
                        add_terrain_noise=False,
                        isRLGymInterface=False,
                        time_step=0.001,
                        action_repeat=1,
                        motor_control_mode="TORQUE",
                        # record_video=True
                        )

  threeLegPose = env._robot_config.INIT_JOINT_ANGLES
  # threeLegPose[3] = -0.4
  env._robot.SetJointStates(np.concatenate((threeLegPose, np.zeros(12))))

  dt = 0.001

  # create time series for thigh, hip, calf joints for wave
  #   hip, 0, 0, then wave back and forth
  #   thigh, go from pi/4 to -1
  hip =  np.array([       0,     0, -0.4,  0.4,         0])
  thigh= np.array([ np.pi/4,    -1,   -1,   -1,   np.pi/4])
  calf = np.array([-np.pi/2,  -0.8, -0.8, -0.8, -np.pi/2] )
  FR_leg = np.vstack((hip, thigh, calf))
  t_arr = np.array([ 0, 1, 2, 3,4])
  t_idx = 0

  # while True:
  #   # increase hip joint 
  #   # FR_thigh = env._robot.GetMotorAngles()[1]
  #   # if FR_thigh > np.pi/4:
  #   #   dt = -0.001
  #   # elif FR_thigh < -1:
  #   #   dt = 0.001
  #   # threeLegPose[0] -= 2*dt
  #   # threeLegPose[1] += dt
  #   # threeLegPose[2] += dt

  #   t = env.get_sim_time()
  #   # find where in time array we are
  #   if t > t_arr[t_idx]:
  #     t_idx += 1
  #   progress = t - t_arr[t_idx]
  #   env.step(threeLegPose)

  # get each foot xyz
  feet_pos = []
  for i in range(4):
    _, pos = env._robot._ComputeJacobianAndPositionUSC(i)
    feet_pos.append(pos)

  FRxyz = feet_pos[0]
  # desired xyz trajectory for FR
  z_off = 0.25
  x_off = 0.2
  FR_xyz_offsets = np.array( [ [0, 0, x_off, x_off-0.02, x_off+0.02, x_off, 0] , 
                               [0, 0,     0,      -0.005,      0.005,     0, 0] ,
                               [0, 0, z_off, z_off-0.02, z_off+0.02, z_off, 0] ])
  # FR_xyz = np.hstack((FRxyz.reshape((3,1)), FR_xyz))

  # test moving body back first, so all legs move forward
  # FR_xyz_offsets = np.hstack((FR_xyz_offsets[:,0:2], ))

  FR_xyz = FRxyz.reshape((3,1)) + FR_xyz_offsets
  # t_arr = np.array([0, 1, 2, 3, 4, 5, 6])
  t_arr = np.array([0, 1, 1.2, 1.3, 1.4, 1.5, 6])
  t_arr = np.array([0, 0.5, 0.65, 0.8, 0.9, 1, 1.5])
  t_idx = 0

  while True:
    action = np.zeros(12)
    for i in range(4):
      if i==0 or i==2:
        FR_hip = np.array([0, -env._robot_config.HIP_LINK_LENGTH, 0])
      else:
        FR_hip = np.array([0, env._robot_config.HIP_LINK_LENGTH, 0])

      t = env.get_sim_time()
      # find where in time array we are
      if t > t_arr[t_idx+1]:
        t_idx += 1
      if i == 0:
        # step through desired xyz trajectory 
        # alpha = (t+1) - t_arr[t_idx]
        alpha = (t) - t_arr[t_idx]
        # alpha = t_arr[t_idx] - t
        # alpha = t_arr[t_idx+1] - t_arr[t_idx]
        # xyz = (1-alpha)*FR_xyz[:,t_idx] + (alpha)*FR_xyz[:,t_idx+1]
        xyz = FR_xyz[:,t_idx] + alpha*(FR_xyz[:,t_idx+1] -FR_xyz[:,t_idx]) / (t_arr[t_idx+1] - t_arr[t_idx])
        print(t,t_idx, alpha, FR_xyz[:,t_idx] , xyz)
      else:
        xyz = feet_pos[i] + min(2*t,1)*np.array([0.06,-0.05,0])
      q_des = env._robot._ComputeIKSpotMicro(i, xyz  ) # - FR_hip
      dq_des = np.zeros(3)

      q = env._robot.GetMotorAngles()[3*i:3*i+3]
      dq = env._robot.GetMotorVelocities()[3*i:3*i+3]
      kps = env._robot_config.MOTOR_KP[3*i:3*i+3]
      kds = env._robot_config.MOTOR_KD[3*i:3*i+3]
      # kps = np.array([50]*3)
      # kds = np.array([0.5]*3)

      tau_joint = kps*(q_des-q) + kds*(dq_des-dq)
      action[3*i:3*i+3] = tau_joint

    env.step(action)

def mainCheckSpherical():
  env = QuadrupedGymEnv(render=True, 
                        on_rack=False,
                        accurate_motor_model_enabled=True,
                        pd_control_enabled=False,
                        randomize_dynamics=True,
                        add_terrain_noise=False,
                        isRLGymInterface=True,
                        time_step=0.001,
                        action_repeat=10,
                        # action_repeat=20,
                        motor_control_mode="IMPEDANCE_POS_ONLY_SPHERICAL_IK",
                        enable_action_filter=True,
                        action_filter_weight=0.1,
                        # record_video=True
                        )

  cntr = -1
  cntr2 = -1
  dt = 0.001
  dt2 = 0.0003

  action_space_size = len(env.action_space.low)
  total_reward = 0
  # while True:
  #   action=2*np.random.random(action_space_size)-1
  #   # action[2::3] = -1
  #   env.step(action)
  #   obs, reward, done, info = env.step(action)
  #   total_reward += reward
  #   if done:
  #     print('reward', total_reward)
  #     total_reward = 0
  #     env.reset()

  while True:
    # print(env._robot.GetBaseOrientation())
    if cntr > 1:
      cntr = -1
    if cntr2 > 1:
      cntr2= 0 
    ang_flip = np.array([1,  1,  1, 1, -1, -1, 
                         1, -1, -1, 1,  1,  1])
    action=2*np.random.random(12) - 1
    # action = -1*np.ones(12) * ang_flip
    # action = cntr*np.ones(12)
    
    # action = np.array([1,cntr,0.6]*4)
    # print(cntr, action)
    cntr += dt
    cntr2 += dt2
    obs, reward, done, info = env.step(action)
    print(env._robot._ComputeJacobianAndPositionUSC(0)[1])
    if done:
      #sys.exit()
      env.reset()


def mainCheckPMTG():
  env = QuadrupedGymEnv(render=True, 
                        on_rack=False,
                        accurate_motor_model_enabled=True,
                        pd_control_enabled=False,
                        randomize_dynamics=False,
                        add_terrain_noise=False,
                        isRLGymInterface=True,
                        time_step=0.001,
                        action_repeat=1,
                        # action_repeat=20,
                        motor_control_mode="PMTG",
                        enable_action_filter=False,
                        action_filter_weight=0.1,
                        # record_video=True
                        )
  total_reward = 0
  action_space_size = len(env.action_space.low)
  while True:
    action=2*np.random.random(action_space_size)-1
    action=np.zeros(12)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
      print('reward', total_reward)
      total_reward = 0
      env.reset()

if __name__ == "__main__":
  # test out some functionalities
  use_gym = True
  test_3_leg = False
  check_spherical = False
  check_pmtg = True
  if check_pmtg:
    mainCheckPMTG()
    sys.exit()

  if test_3_leg:
    main3leg()
    sys.exit()
  if check_spherical:
    mainCheckSpherical()
    sys.exit()

  # NOT using gym, test spherical coordinates
  # r will be in [ 0.15, 0.35]
  # theta (from x to y) can be in [-pi,pi]
  # psi (away from z) should really be different based on if in x or y direction
  #     try [-0.8,0.8] due to hip joint requirement
  def spherical2cartesian(r,theta,psi):
    x = r*np.sin(theta)*np.cos(psi)
    y = r*np.sin(theta)*np.sin(psi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

  def unscale_helper(action, lower_lim, upper_lim):
    """Helper to linearly scale from action to  [-1,1] . 
    This needs to be made general in case action range is not [-1,1]
    """
    unscaled_action = 2 * ( (action - lower_lim)) /  (upper_lim - lower_lim) - 1 
    # new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
    # verify clip
    new_a = np.clip(unscaled_action, -1, 1)
    return new_a

  env = QuadrupedGymEnv(render=True, 
                        on_rack=False,
                        accurate_motor_model_enabled=True,
                        pd_control_enabled=False,
                        randomize_dynamics=False,
                        add_terrain_noise=False,
                        isRLGymInterface=False,
                        time_step=0.001,
                        action_repeat=1,
                        motor_control_mode="TORQUE",
                        enable_action_filter=False,
                        # record_video=True
                        )


  psi_upp = 0.8
  dt = 0.001
  psi = -psi_upp

  r_upp = 0.35
  r_low = 0.15
  r = r_low

  ftg = FootTrajectoryGenerator(T=0.5,max_foot_height=0.1)

  for counter in range(20000):
    
    # if counter % 50 == 0:
    psi += dt
    r += dt
    if psi > psi_upp:
      psi = -psi_upp

    if r > r_upp:
      r = r_low
    # for _ in range(4):
    foot_dh = ftg.setPhasesAndGetDeltaHeights(env.get_sim_time(),fi=np.zeros(4))
    print('==========================')
    action = np.zeros(12)
    for i in range(4):
      if i==0 or i==2:
        theta = -psi
      else:
        theta = -psi
      # foot_dh = self._ftg.setPhasesAndGetDeltaHeights(self.get_sim_time(), fi=fi)
      # foot_dh = ftg.setPhasesAndGetDeltaHeights(env.get_sim_time(), fi=np.zeros(4))

      # r = 0.2+foot_dh[i]
      r = 0.3-foot_dh[i]
      
      # if i==2 or i==3:
      #   r = 0.2

      spherical_coords = spherical2cartesian(r,0,0)
      # just test FR for now
      if i==0 or i==2:
        FR_hip = np.array([0, -env._robot_config.HIP_LINK_LENGTH, 0])
      else:
        FR_hip = np.array([0, env._robot_config.HIP_LINK_LENGTH, 0])

      # move feet inwards
      in_factor = 0.03
      if i==0 or i==2:
        FR_hip += np.array([0, in_factor, 0])
      else:
        FR_hip += np.array([0, -in_factor, 0])

      xyz = np.array([1,1,-1])*spherical_coords + FR_hip
      # print(xyz)
      kpCartesian = env._robot_config.kpCartesian
      kdCartesian = env._robot_config.kdCartesian
      kpCartesian = np.diag([3000]*3)
      kdCartesian = np.diag([50]*3)
      # kpCartesian = np.diag([1000]*3)
      
      # kpCartesian = np.diag([500]*3)
      # kdCartesian = np.diag([10]*3)
      kpCartesian = np.diag([800]*3)
      kdCartesian = np.diag([15]*3)

      FR_tau = env._robot._ComputeLegImpedanceControlActions(xyz,
                                              np.zeros(3),
                                              kpCartesian,
                                              kdCartesian,
                                              np.zeros(3), # ,#np.array([0,0,-30]),#
                                              i)
      action[3*i:3*i+3] = FR_tau

      ##################################
      # test with adding joint PD as well 
      # current angles
      q = env._robot.GetMotorAngles()[3*i:3*i+3]
      dq = env._robot.GetMotorVelocities()[3*i:3*i+3]
      kps = env._robot_config.MOTOR_KP[3*i:3*i+3]
      kds = env._robot_config.MOTOR_KD[3*i:3*i+3]

      # test other kps
      kps = np.array([50]*3)
      kds = np.array([0.5]*3)
      # # kps = np.array([35]*3)
      # kds = np.array([1]*3)

      # kps = np.array([27]*3)
      # kds = np.array([0.3]*3)

      # # 1000, 15, 50, 1
      # kps = np.array([20]*3)
      # kds = np.array([0.2]*3)

      # corresponding desired joint angles:
      q_des = env._robot._ComputeIKSpotMicro(i, xyz ) # - FR_hip
      dq_des = np.zeros(3)
      tau_joint = kps*(q_des-q) + kds*(dq_des-dq)

      action[3*i:3*i+3] += tau_joint

    # action = np.concatenate((xyz,np.zeros(9)))
    env.step(action)

    # print('='*50)
    # for i in range(4):
    # print(env._robot.GetLocalEEPos(i))
    print(psi,xyz, env._robot._ComputeJacobianAndPositionUSC(0)[1])
    # time.sleep(0.1)

    # if done:
    #   env.reset()

  env.reset()