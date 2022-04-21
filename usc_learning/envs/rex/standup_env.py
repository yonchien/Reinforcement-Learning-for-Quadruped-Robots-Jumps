"""This file implements the gym environment of rex alternating legs.

"""
import math
import random

from gym import spaces
import numpy as np
from usc_learning.envs.rex import rex_gym_env
from usc_learning.envs.rex.rex import Rex

STEP_PERIOD = 1.0 / 15.0  # 15 steps per second.
STEP_AMPLITUDE = 0.25

NUM_LEGS = 4
NUM_MOTORS = 3 * NUM_LEGS


class RexStandupEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.

  It simulates the locomotion of a rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the rex walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}

    def __init__(self,
                 urdf_version=None,
                 control_time_step=0.006,
                 action_repeat=6,
                 control_latency=0,
                 pd_latency=0,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 remove_default_joint_damping=False,
                 render=False,
                 num_steps_to_log=1000,
                 env_randomizer=None,
                 log_path=None):
        """Initialize the rex alternating legs gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. Refer to
        rex_gym_env for more details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See minituar.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See rex.py for more details.
      on_rack: Whether to place the rex on rack. This is only used to debug
        the walking gait. In this mode, the rex's base is hung midair so
        that its walking gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
        randomize the environment during when env.reset() is called and add
        perturbations when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        rex_logging.proto.
    """
        super(RexStandupEnv,
              self).__init__(urdf_version=urdf_version,
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             hard_reset=False,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=remove_default_joint_damping,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat)

        action_dim = 12
        action_high = np.array([0.1] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 30
        self._cam_pitch = -30
        self.stand = False
        if self._on_rack:
            self._cam_pitch = 0

    def reset(self):
        self.desired_pitch = 0
        super(RexStandupEnv, self).reset(initial_motor_angles=Rex.INIT_POSES['rest_position'],
                                         reset_duration=0.5)
        return self._get_observation()

    def _signal(self, t):
        if t > 0.2:
            self.stand = True
            return self.rex.INIT_POSES['stand_low']
        t += 1
        # apply a 'brake' function
        signal = self.rex.INIT_POSES['stand_low'] * (0.1 / t + 1.5)
        return signal

    def _convert_from_leg_model(self, leg_pose):
        if self.stand:
            return self.rex.INIT_POSES['stand_low']
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[3 * i] = leg_pose[3 * i]
            motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
            motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
        return motor_pose

    def _transform_action_to_motor_command(self, action):
        action += self._signal(self.rex.GetTimeSinceReset())
        action = self._convert_from_leg_model(action)
        return action

    def _termination(self):
        # if self.is_fallen():
        #     print("IS FALLEN!")
        return self.is_fallen()

    def is_fallen(self):
        """Decide whether the rex has fallen.

    If the up directions between the base and the world is large (the dot
    product is smaller than 0.85), the rex is considered fallen.

    Returns:
      Boolean value that indicates whether the rex has fallen.
    """
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        return math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.5

    def _reward(self):
        # target position
        t_pos = [0.0, 0.0, 0.225]

        current_base_position = self.rex.GetBasePosition()

        position_reward = abs(t_pos[0] - current_base_position[0]) + \
                          abs(t_pos[1] - current_base_position[1]) + \
                          abs(t_pos[2] - current_base_position[2])

        is_pos = False

        if abs(position_reward) < 0.1:
            position_reward = 1.0 - position_reward
            is_pos = True
        else:
            position_reward = -position_reward

        if current_base_position[2] > t_pos[2]:
            position_reward = -1000 - position_reward
            #print("jump!")

        if is_pos:
            self.goal_reached = True

        reward = position_reward
        return reward

    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, the error between current pitch and desired pitch,
    roll dot and pitch dot of the base.

    Returns:
      The observation list.
    """
        observation = []
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetTrueBaseRollPitchYawRate()
        observation.extend([roll, pitch, roll_rate, pitch_rate])
        observation[1] -= self.desired_pitch  # observation[1] is the pitch
        self._true_observation = np.array(observation)
        return self._true_observation

    def _get_observation(self):
        observation = []
        roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
        observation.extend([roll, pitch, roll_rate, pitch_rate])
        observation[1] -= self.desired_pitch  # observation[1] is the pitch
        self._observation = np.array(observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.zeros(self._get_observation_dimension())
        upper_bound[0:2] = 2 * math.pi  # Roll, pitch, yaw of the base.
        upper_bound[2:4] = 2 * math.pi / self._time_step  # Roll, pitch, yaw rate.
        return upper_bound

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound

    def set_swing_offset(self, value):
        """Set the swing offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
        self._swing_offset = value

    def set_extension_offset(self, value):
        """Set the extension offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
        self._extension_offset = value

    def set_desired_pitch(self, value):
        """Set the desired pitch of the base, which is a user input.

    Args:
      value: A scalar.
    """
        self.desired_pitch = value
