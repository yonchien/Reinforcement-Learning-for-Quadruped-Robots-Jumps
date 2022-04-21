"""This file implements the gym environment of a quadruped. 
Subclass as necessary. CLEAN UP from quadruped_gym_env.py, have robot config functionality

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

import usc_learning.envs.pyb_data as pyb_data
from usc_learning.envs.my_minitaur import my_minitaur_env_randomizer
import usc_learning.envs.quadruped_master_old.quadruped as quadruped
from pkg_resources import parse_version

LAIKAGO_FLAG = False
#RECORD_VIDEO = False

if LAIKAGO_FLAG:
  import usc_learning.envs.quadruped_master.configs_laikago as robot_config
else:
  import usc_learning.envs.quadruped_master.configs_aliengo as robot_config

#import usc_learning.envs.quadruped_master.configs_a1 as robot_config


ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
#RENDER_HEIGHT = 720
#RENDER_WIDTH = 960

VIDEO_LOG_DIRECTORY = 'videos/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")

""" Implemented task environemnts """
TASK_ENVS = ["DEFAULT", "FWD_LOCOMOTION", "STANDUP",  "STAYDOWN"]
# default == fw_locomotion

CONTROL_MODES = [ "TORQUE", "TORQUE_NO_HIPS", "PD", "PD_NO_HIPS", "IK", "FORCE", "IMPEDANCE", "IMPEDANCE_FORCE" ]
# IMPEDANCE_FORCE: pre-determined desired feet locations, just choose 4 normal forces

class QuadrupedGymEnv(gym.Env):
  """The gym environment for a quadruped.

  It simulates the locomotion of a quadrupedal robot. The state space
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
      time_step=0.01,
      action_repeat=1,
      distance_weight=1.0,
      energy_weight=0.005,
      shake_weight=0.0,
      drift_weight=0.0,
      distance_limit=float("inf"),
      observation_noise_stdev=0.0,
      self_collision_enabled=True,
      motor_velocity_limit=np.inf,
      pd_control_enabled=False,  #not needed to be true if accurate motor model is enabled (has its own better PD)
      leg_model_enabled=True,
      accurate_motor_model_enabled=True,
      motor_control_mode="IMPEDANCE",
      task_env="FWD_LOCOMOTION",
      motor_overheat_protection=True,
      hard_reset=True,
      on_rack=False,
      render=False,
      record_video=False,
      env_randomizer=None): #my_minitaur_env_randomizer.MinitaurEnvRandomizer()):
    """Initialize the minitaur gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
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
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_control_mode: Whether to use torque control, PD, control, etc.
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
      env_randomizer: An EnvRandomizer to randomize the physical properties
        during reset().
    """
    self._robot_config = robot_config
    self._isRLGymInterface = isRLGymInterface
    self._time_step = time_step
    self._action_repeat = action_repeat
    self._num_bullet_solver_iterations = 300
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
    self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._motor_control_mode = motor_control_mode
    self._TASK_ENV = task_env
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    # self._cam_dist = 1.0
    # self._cam_yaw = 0
    # self._cam_pitch = -30
    self._configure_visualizer()
    self._hard_reset = True
    self._last_frame_time = 0.0
    print("urdf_root=" + self._urdf_root)
    self._env_randomizer = env_randomizer
    # PD control needs smaller time step for stability.
    if pd_control_enabled or accurate_motor_model_enabled:
      self._time_step /= self.NUM_SUBSTEPS
      self._num_bullet_solver_iterations /= self.NUM_SUBSTEPS
      self._action_repeat *= self.NUM_SUBSTEPS

    if not isRLGymInterface:
      self._time_step = time_step
      self._action_repeat = action_repeat
      self._num_bullet_solver_iterations = 300

    if self._is_render:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bc.BulletClient()

    self.videoLogID = None
    self.seed()
    self.reset()
    # observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
    # observation_low = (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)
    # if motor_control_mode in ["PD","TORQUE"]:
    #   action_dim = 12
    # elif motor_control_mode in ["IMPEDANCE"]:
    #   action_dim = 16
    # elif motor_control_mode in ["IMPEDANCE_FORCE"]:
    #   action_dim = 4
    # else:
    #   raise ValueError("motor control mode" + motor_control_mode + "not implemented yet.")
    # self._action_bound = 1
    # action_high = np.array([self._action_bound] * action_dim)
    # self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    # self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
    self.setupObsAndActionSpaces()
    self._hard_reset = hard_reset  # This assignment need to be after reset()

    self._MAX_EP_LEN = 5 # max sim time in seconds, arbitrary

  def setupObsAndActionSpaces(self):
    """Set up observation and action spaces for RL. """
    observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
    observation_low = (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)
    if self._motor_control_mode in ["PD","TORQUE"]:
      action_dim = 12
    elif self._motor_control_mode in ["IMPEDANCE"]:
      action_dim = 16
    elif self._motor_control_mode in ["IMPEDANCE_FORCE"]:
      action_dim = 4
    else:
      raise ValueError("motor control mode" + motor_control_mode + "not implemented yet.")
    self._action_bound = 1
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
    

  ######################################################################################
  # Reset
  ######################################################################################
  def reset(self):
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
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
      self._robot.Reset(reload_urdf=False)

    if self._env_randomizer is not None:
      raise ValueError('Not implemented randomizer yet.')
      self._env_randomizer.randomize_env(self)

    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
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
    #if self._motor_control_mode is not "TORQUE":
    for _ in range(200):
      if self._pd_control_enabled or self._accurate_motor_model_enabled:
        #self._robot.ApplyAction([math.pi / 2] * 8)
        self._robot.ApplyAction(self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS)
        #print('Motor angles', self._robot.GetMotorAngles())
        #print('Motor torques', self._robot.GetMotorTorques())
      self._pybullet_client.stepSimulation()
      # if self._is_render:
        #   time.sleep(0.1)
    # set control mode back
    self._motor_control_mode = tmp_save_motor_control_mode_ENV
    self._robot._motor_control_mode = tmp_save_motor_control_mode_ROB
    self._robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT

    if self._is_record_video:
      self.recordVideoHelper()
    return self._noisy_observation()


  ######################################################################################
  # Step 
  ######################################################################################
  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError("{}th action {} out of bounds.".format(i, action_component))
      """ CHANGED """
      #action = self._robot.ConvertFromLegModel(action)
      if self._motor_control_mode == "PD":
        action = self._robot.ScaleActionToPosRange(action)
      elif self._motor_control_mode in ["IMPEDANCE", "IMPEDANCE_FORCE"]:
        action = self._robot.ScaleActionToImpedanceControl(action)
      elif self._motor_control_mode is not "TORQUE":
        # default to torque (no scaling)
        raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
    return action

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    if self._is_render:
      self._render_step_helper()
    # only scale action if it's a gym interface
    
    for _ in range(self._action_repeat):
      if self._isRLGymInterface:
        proc_action = self._transform_action_to_motor_command(action)
      else:
        proc_action = action
      self._robot.ApplyAction(proc_action)
      self._pybullet_client.stepSimulation()
      self._sim_step_counter += 1

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination() or self.get_sim_time() > self._MAX_EP_LEN
    return np.array(self._noisy_observation()), reward, done, {}

  ######################################################################################
  # Termination and reward
  ######################################################################################
  def is_fallen(self):
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
    #if LAIKAGO_FLAG: # taller than minitaur
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < self._robot_config.IS_FALLEN_HEIGHT)
    # else:
    #   # default for minitaur 
    #   return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < 0.13)

  def _termination(self):
    position = self._robot.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self.is_fallen() or distance > self._distance_limit

  def _reward_orig(self):
    """ Reward that worked originally for locomotion task. """
    current_base_position = self._robot.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
    self._last_base_position = current_base_position
    energy_reward = np.abs(
        np.dot(self._robot.GetMotorTorques(),
               self._robot.GetMotorVelocities())) * self._time_step
    reward = (self._distance_weight * forward_reward - self._energy_weight * energy_reward +
              self._drift_weight * drift_reward + self._shake_weight * shake_reward)
    self._objectives.append([forward_reward, energy_reward, drift_reward, shake_reward])
    return reward

  def _reward_fwd_locomotion(self):
     # added survival bonus to original reward
    return self._reward_orig() #+ 0.1

  def _reward_stand(self, target_height=0.4):
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
    reward= .4 - abs(target_height-curr_z)

    #return reward - self._energy_weight * energy_reward - 0.1*orn_reward 
    return reward #- 0.001 * motor_vel_reward #- 0.1*orn_reward 

  def _reward(self):
    """ Get reward depending on task"""
    if self._TASK_ENV == "DEFAULT":
      return self._reward_orig()
    elif self._TASK_ENV == "STANDUP":
      return self._reward_stand()
    elif self._TASK_ENV == "FWD_LOCOMOTION":
      return self._reward_fwd_locomotion()
    else:
      raise ValueError("This task mode not implemented yet.")


  ######################################################################################
  # Observation
  ######################################################################################
  def get_objectives(self):
    return self._objectives

  def _get_observation(self):
    self._observation = self._robot.GetObservation()
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += (
          np.random.normal(scale=self._observation_noise_stdev, size=observation.shape) *
          self._robot.GetObservationUpperBound())
    return observation

  def get_sim_time(self):
    """ Get current simulation time from pybullet robot. """
    #return self._env_step_counter * self._action_repeat * self._time_step
    return self._sim_step_counter * self._time_step

  ######################################################################################
  # Render and misc
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

  def recordVideoHelper(self):
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
    time_to_sleep = self._action_repeat * self._time_step - time_spent
    if time_to_sleep > 0:
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

  def _configure_visualizer(self):
    """ Remove all visualizer borders, and zoom in """
    # default rendering options
    self._render_width = 960
    self._render_height = 720
    # self._camera_dist = 1
    # self._camera_pitch = -30
    # self._camera_yaw = 0
    self._cam_dist = 1.0
    self._cam_yaw = 0
    self._cam_pitch = -30
    # get rid of random visualizer things
    # self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
    # self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
    # self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
    # self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)

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

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
