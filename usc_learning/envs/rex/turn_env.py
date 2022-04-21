"""This file implements the gym environment of rex alternating legs.

"""
import math
import random

from gym import spaces
import numpy as np
from usc_learning.envs.rex import rex_gym_env

NUM_LEGS = 4
NUM_MOTORS = 3 * NUM_LEGS
STEP_PERIOD = 1.0 / 10.0  # 10 steps per second.
TARGET_POSITION = [0.0, 0.0, 0.21]


class RexTurnEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.
    It simulates the locomotion of a rex, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the rex walks in 1000 steps and penalizes the energy expenditure."""
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
                 log_path=None,
                 target_orient=None,
                 init_orient=None):
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
        super(RexTurnEnv, self).__init__(
            urdf_version=urdf_version,
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
            action_repeat=action_repeat,
            target_orient=target_orient,
            init_orient=init_orient)

        action_dim = 12
        action_high = np.array([0.1] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 30
        self._cam_pitch = -30
        self.last_step = 0

        self.stand = False
        self.brake = False
        self._target_orient = target_orient
        self._init_orient = init_orient
        self._random_target = False
        self._random_start = False
        if self._on_rack:
            self._cam_pitch = 0

    def reset(self):
        self.desired_pitch = 0
        self.goal_reached = False
        self.stand = False
        super(RexTurnEnv, self).reset()
        if self._target_orient is None or self._random_target:
            self._target_orient = random.uniform(0.2, 6)
            self._random_target = True

        if self._on_rack:
            # on rack debug simulation
            self._init_orient = 2.1
            position = self.rex.init_on_rack_position
        else:
            position = self.rex.init_position
            if self._init_orient is None or self._random_start:
                self._init_orient = random.uniform(0.2, 6)
                self._random_start = True
        print(f"Start Orientation: {self._init_orient}, Target Orientation: {self._target_orient}")
        print("Turning left") if self._init_orient - self._target_orient > 3.14 else print("Turning right")
        q = self.pybullet_client.getQuaternionFromEuler([0, 0, self._init_orient])
        self.pybullet_client.resetBasePositionAndOrientation(self.rex.quadruped, position, q)
        return self._get_observation()

    def _signal(self, t):
        initial_pose = self.rex.INIT_POSES['stand_high']
        period = STEP_PERIOD
        extension = 0.1
        swing = 0.2
        swipe = 0.1
        ith_leg = int(t / period) % 2

        pose = {
            'left_0': np.array([swipe, extension, -swing,
                                -swipe, extension, swing,
                                swipe, -extension, swing,
                                -swipe, -extension, -swing]),
            'left_1': np.array([-swipe, 0, swing,
                                swipe, 0, -swing,
                                -swipe, 0, -swing,
                                swipe, 0, swing]),
            'right_0': np.array([-swipe, extension, -swing,
                                 swipe, -extension, swing,
                                 -swipe, -extension, swing,
                                 swipe, -extension, -swing]),
            'right_1': np.array([swipe, 0, swing,
                                 -swipe, 0, -swing,
                                 swipe, 0, -swing,
                                 -swipe, 0, swing])
        }
        diff = self._init_orient - self._target_orient
        if diff > 3.14 or diff < 0:
            # turn left
            first_leg = pose['left_0']
            second_leg = pose['left_1']
        else:
            # turn right
            first_leg = pose['right_0']
            second_leg = pose['right_1']

        if ith_leg:
            signal = initial_pose + second_leg
        else:
            signal = initial_pose + first_leg
        return signal

    def _convert_from_leg_model(self, leg_pose, t):
        if t < .4:
            # set init position
            return self.rex.INIT_POSES['stand_high']
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            if self.goal_reached:
                return self.rex.initial_pose
            else:
                motor_pose[3 * i] = leg_pose[3 * i]
                motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
                motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
        return motor_pose

    def _transform_action_to_motor_command(self, action):
        t = self.rex.GetTimeSinceReset()
        action += self._signal(t)
        action = self._convert_from_leg_model(action, t)
        return action

    def is_fallen(self):
        """Decide whether the rex has fallen.

        If the up directions between the base and the world is large (the dot
        product is smaller than 0.85), the rex is considered fallen.

        Returns:
          Boolean value that indicates whether the rex has fallen.
        """
        orientation = self.rex.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def _reward(self):
        current_base_position = self.rex.GetBasePosition()
        current_base_orientation = self.pybullet_client.getEulerFromQuaternion(self.rex.GetBaseOrientation())
        target_orient = (0, 0, self._target_orient)
        proximity_reward = \
            abs(target_orient[0] - current_base_orientation[0]) + \
            abs(target_orient[1] - current_base_orientation[1]) + \
            abs(target_orient[2] - current_base_orientation[2])

        position_reward = \
            abs(TARGET_POSITION[0] - current_base_position[0]) + \
            abs(TARGET_POSITION[1] - current_base_position[1]) + \
            abs(TARGET_POSITION[2] - current_base_position[2])

        is_oriented = False
        is_pos = False
        if abs(proximity_reward) < 0.1:
            proximity_reward = 100 - proximity_reward
            is_oriented = True
        else:
            proximity_reward = -proximity_reward

        if abs(position_reward) < 0.1:
            position_reward = 100 - position_reward
            is_pos = True
        else:
            position_reward = -position_reward

        if is_pos and is_oriented:
            self.goal_reached = True
            self.goal_t = self.rex.GetTimeSinceReset()

        reward = position_reward + proximity_reward
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
