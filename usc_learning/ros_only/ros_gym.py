""" ROS gym, to be used with RL """
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from laikago_msgs.msg import LowState, MotorState, LowCmd, MotorCmd
from laikago_msgs.msg import RLFootPos
from gazebo_msgs.msg import ModelStates

from std_srvs.srv import Empty, EmptyRequest
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Int32

import numpy as np
import time
from datetime import datetime

from usc_learning.ros_only.gazebo_comm import GazeboComm
from usc_learning.ros_only.current_state import CurrentState, lowState
from usc_learning.ros_only.leg_controller import LegController, lowCmd
from usc_learning.ros_only.quadruped_ros import Quadruped

import usc_learning.envs.quadruped_master.configs_a1 as robot_config

# gym
import gym
from gym import spaces

# RL setup
OBSERVATION_EPS = 0.01


rospy.init_node('ros_gym', anonymous=True)

# task modes
"""
either 
1) MPC_FOOT_POS: choose foot positions for MPC
   MPC_FOOT_POS_VEL: choose foot positions for MPC, also desired forward velocity
   MPC_VEL: choose high level velocity only


2) BALANCE_NO_FALL: use Balance Controller to not fall down during training
    -i.e. takes over when detecting fall

"""

class ROSGymEnv(gym.Env):
    """ Pure ROS gym env, uses gazebo comm, current_state, leg_controller, etc. 
    Check mode! First test: mpc adjust RL position
    """

    def __init__(self,
            robot_config=robot_config,
            isRLGymInterface=True,
            #added
            distance_weight=1.0,
            energy_weight=0.1,#0.005,
            shake_weight=0.0,
            drift_weight=0.0,
            distance_limit=float("inf"),
            observation_noise_stdev=0.0,
            #
            task_mode="MPC_VEL",
            accurate_motor_model_enabled=True,
            motor_control_mode="IMPEDANCE",
            task_env="FWD_LOCOMOTION",
            observation_space_mode="DEFAULT_CONTACTS",
            rl_action_high=1.0,       # action high for RL
            grow_action_space=False,  # use growing action spaces
            #hard_reset=True,
            #render=True,
            #record_video=False,
            env_randomizer=None,
            randomize_dynamics=False,
            add_terrain_noise=False,
            **kwargs
        ):
        self._task_mode = task_mode
        self._motor_control_mode = motor_control_mode
        self._observation_space_mode = observation_space_mode
        self._GAS = grow_action_space
        self._robot_config = robot_config
        self._observation_noise_stdev = observation_noise_stdev
        self.num_motors = 12
        self._action_bound = 1

        self._robot = Quadruped(robot_config)
        #self._gazebo_comm = GazeboComm()
        self._setup_ros_comm()
        self.reset()
        self.setupObsAndActionSpaces()


        self._last_base_position = [0, 0, 0]
        self._distance_weight = distance_weight
        self._energy_weight = energy_weight
        self._drift_weight = drift_weight
        self._shake_weight = shake_weight
        self._distance_limit = distance_limit

        self._MAX_EP_LEN = 30


    def setupObsAndActionSpaces(self):
        """Set up observation and action spaces for RL. """
        # Observation space
        if self._observation_space_mode == "DEFAULT":
            raise NotImplementedError
            observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
            observation_low =  (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)
        elif self._observation_space_mode == "DEFAULT_CONTACTS":
            observation_high = (self.GetObservationContactsUpperBound() + OBSERVATION_EPS)
            observation_low =  (self.GetObservationContactsLowerBound() - OBSERVATION_EPS)
        elif self._observation_space_mode == "EXTENDED":
            raise NotImplementedError
            observation_high = (self._robot.GetExtendedObservationUpperBound() + OBSERVATION_EPS)
            observation_low =  (self._robot.GetExtendedObservationLowerBound() - OBSERVATION_EPS)
        else:
            raise ValueError("observation space not defined or not intended")
        # Action space
        # if self._motor_control_mode in ["IMPEDANCE_XZ_ONLY"]:
        #     action_dim = 8

        if self._task_mode in ['MPC_FOOT_POS']:
            action_dim = 8 # 4 xy positions
        elif self._task_mode in ['MPC_FOOT_POS_VEL']:
            action_dim = 9 # 4 xy positions, 1 high level vel
        elif self._task_mode == "MPC_VEL":
            action_dim = 1 # 1 high level vel
        elif self._motor_control_mode in ["PD","TORQUE", "IK", "IMPEDANCE_NO_Y"]:
          action_dim = 12
        elif self._motor_control_mode in ["IMPEDANCE"]:
          action_dim = 16 # (xyz pos + Fn)*4
        # elif self._motor_control_mode in ["IMPEDANCE_FORCE"]:
        #   action_dim = 4 # Fn*4
        # elif self._motor_control_mode in ["IMPEDANCE_GAINS"]:
        #   action_dim = 22 # (xyz pos + Fn)*4 + 3 kp_gains + 3 kd_gains
        # elif self._motor_control_mode in ["IMPEDANCE_3D_FORCE"]:
        #   action_dim = 24 # (xyz pos + 3D contact force)*4  
        else:
          raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")

        if self._GAS: # likely in lower range than [-1,1]
            action_high = np.array([self._rl_action_bound] * action_dim)
        else:
            action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

    def reset(self):
        """ Reset, call super and reset any necessary MPC/balance controller 

        (Mostly) everything ish handled in the C++ rl_gym_helper class
        """
        # self._gazebo_comm.resetSim()
        # self._gazebo_comm.resetWorld()
        # self._robot.reset()
        # call reset again?
        #self._gazebo_comm.resetWorld()

        # reset any MPC or balance controller needed things
        # print('Reset after ',self.get_sim_time(), '(s)')
        # try:
        #     print( 'pos:', self._robot.GetBasePosition())
        # except:
        #     pass
        #self.resetMPC()
        #self.resetBalanceController()

        self.reset_env_service(EmptyRequest())

        while rospy.get_time() < 2: # wait for it to stabilize
            time.sleep(0.1)

        #time.sleep(0.5)

        self._last_base_position = self._robot.currentState.body_pos

        #noisy_obs = super(ROSQuadrupedGymEnv, self).reset()
        #self._starting_ee_pos = np.concatenate(([self._robot.GetLocalEEPos(i) for i in range(4)]))
        return self._noisy_observation() #noisy_obs #super(ROSQuadrupedGymEnv, self).reset()

    def resetMPC(self):
        """Reset MPC """
        self._resetMPC_pub.publish(Empty())

    def _setup_ros_comm(self):
        """Setup and necessary ROS comm. """
        # reset MPC
        self._resetMPC_pub = rospy.Publisher("/laikago_gazebo/reset_mpc", EmptyMsg, queue_size=1)
        # set gait
        self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)
        # msg containing v_des, yaw_rate_des, rlFootPos
        self._setRLActions_pub = rospy.Publisher("/laikago_gazebo/set_rl_foot_pos", RLFootPos, queue_size=1)

        rospy.wait_for_service('/laikago_gazebo/reset_env_service')
        try:
            self.reset_env_service = rospy.ServiceProxy('/laikago_gazebo/reset_env_service', Empty, persistent=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        # create msgs
        #self.rlFootPos = RLFootPos()

    def setGait(self,num):
        """ Set gait num for MPC: 
        1 bounding
        2 trotting
        3 walking
        4 pacing
        5 galloping
        6 pronking
        """
        print('Setting gait', num)
        data = Int32()
        data.data = int(num)
        if num > 0 and num < 7:
            self._setGait_pub.publish(data)
        else:
            raise ValueError(num + 'is not a valid gait number.')


    ##########################################################################################
    # Step
    ##########################################################################################
    def _transform_action_to_motor_command(self, action):
        """Transform action, depending on mode. """
        if "MPC" in self._task_mode:
            #print('TODO: scale action to necessary position limits')
            action = self._ScaleActionToFootPosRange(action)
        # elif self._motor_control_mode == "PD":
        #     action = self._robot.ScaleActionToPosRange(action)
        # elif self._motor_control_mode == "IK":
        #     action = self._robot.ScaleActionToIKRange(action)
        # elif self._motor_control_mode in ["IMPEDANCE", "IMPEDANCE_FORCE","IMPEDANCE_GAINS","IMPEDANCE_3D_FORCE"]:
        #     action, _ = self._robot.ScaleActionToImpedanceControl(action)
        # elif self._motor_control_mode == "TORQUE":
        #     action = self._robot.ScaleActionToTorqueRange(action)
        else:
            raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
        return action

    def _set_action_ros(self,action):
        """Helper to get action from ROS (if needed) """
        if "MPC" in self._task_mode: # == "MPC_FOOT_POS":
            return self.sendFootPositions(action)
        else:
            raise ValueError('invalid task mode')

    def sendFootPositions(self,action):
        data = RLFootPos()
        if len(action) == 1:
            data.velFlag = 1
            data.v_des[0] = action[0]
            data.yawRateFlag = 0
            for i in range(8):
                data.rlFootPos[i] = 0.
            self._setRLActions_pub.publish(data)
            return

        elif len(action) == 9:
            data.velFlag = 1 
            data.v_des[0] = action[8]
        else:
            data.velFlag = 0
        data.yawRateFlag = 0
        for i in range(8):
            data.rlFootPos[i] = action[i]

        self._setRLActions_pub.publish(data)

    def step(self,action):

        # transform action
        proc_action = self._transform_action_to_motor_command(action)
        # process in ros action
        self._set_action_ros(proc_action)

        # wait (0.03 s?)
        time.sleep(0.1)

        # get reward, obs
        
        reward = self._reward()
        done = self._termination()
        return np.array(self._noisy_observation()), reward, done, {}

    def _scale_helper(self, action, lower_lim, upper_lim):
        """Helper to linearly scale from [-1,1] to lower/upper limits. 
        This needs to be made general in case action range is not [-1,1]
        """
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        # verify clip
        new_a = np.clip(new_a, lower_lim, upper_lim)
        return new_a

    def _ScaleActionToFootPosRange(self,actions):
        """Scale actions to foot position ranges """
        #print('Check what relative positions MPC is using')
        if len(actions) == 1:
            low_lim = 0
            upp_lim = 2
        elif len(actions) == 9:
            # choosing velocity as well
            #low_lim =  np.concatenate((self._robot_config.IK_POS_LOW_MPC, [0]))
            #upp_lim =  np.concatenate((self._robot_config.IK_POS_UPP_MPC, [1]))
            low_lim =  np.concatenate(([-0.05]*8, [0]))
            upp_lim =  np.concatenate(([ 0.05]*8, [2]))
        else:
            low_lim = -.05 #self._robot_config.IK_POS_LOW_MPC
            upp_lim =  .05 #self._robot_config.IK_POS_UPP_MPC

        u = np.clip(actions,-1,1)
        u = self._scale_helper(u, low_lim, upp_lim) 
        # print('low_lim',low_lim)
        # print('upp lim', upp_lim)
        # print(u)
        return u

    def _reward_orig(self):
        """ Reward that worked originally for locomotion task. 
        i.e. how far the quadruped walks (x velocity) while penalizing energy
        """
        current_base_position = self._robot.currentState.body_pos
        forward_reward = current_base_position[0] - self._last_base_position[0]
        drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
        shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
        self._last_base_position = current_base_position

        """ Not clear what reward should really be in this case - just forward locomotion probably """
        energy_reward = 0
        # energy_reward = np.abs(
        #     np.dot(self._robot.current_state.joint_tau, #,
        #            self._robot.current_state.joint_vel)) * self._time_step
        reward = (self._distance_weight * forward_reward - self._energy_weight * energy_reward +
                  self._drift_weight * drift_reward + self._shake_weight * shake_reward)
        #self._objectives.append([forward_reward, energy_reward, drift_reward, shake_reward])
        return reward

    def _reward_fwd_locomotion(self):
        """ Add survival bonus to original reward (?) """
        return self._reward_orig() + .1

    def _reward(self):
        return self._reward_fwd_locomotion()

    def _termination(self):
        """Should terminate episode """
        return self._robot.currentState.body_pos[2] < self._robot_config.IS_FALLEN_HEIGHT \
            or  rospy.get_time() > self._MAX_EP_LEN # max sim time in seconds, arbitrary # over some sim time? 

    ##########################################################################################
    # Observation
    ##########################################################################################

    def _get_observation(self):
        """Get observation, depending on obs space selected. """
        # if self._observation_space_mode == "DEFAULT":
        #     self._observation = np.array(self._robot.GetObservation())
        #     self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
        #     self._robot.GetObservationUpperBound())
        if self._observation_space_mode == "DEFAULT_CONTACTS":
            self._observation = np.array(self.GetObservationContacts())
            self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
            self.GetObservationContactsUpperBound())
        # elif self._observation_space_mode == "EXTENDED":
        #     self._observation = np.array(self._robot.GetExtendedObservation())
        #     self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
        #     self._robot.GetExtendedObservationUpperBound())
        else:
            raise ValueError("observation space not defined or not intended")
        #self._observation = self._robot.GetObservation()
        return self._observation

    def _noisy_observation(self):
        self._get_observation()
        observation = np.array(self._observation)
        if self._observation_noise_stdev > 0:
            observation += self._add_obs_noise
        return observation

    def GetObservationContacts(self):
        """ Use CurrentState for observation, will need to update with actual robot states """
        observation = []
        observation.extend(self._robot.currentState.joint_pos.tolist())
        observation.extend(self._robot.currentState.joint_vel.tolist())
        observation.extend(self._robot.currentState.joint_tau.tolist())
        observation.extend(list(self._robot.currentState.body_orn))
        observation.extend([self._robot.currentState.body_pos[2]])
        observation.extend(self._robot.currentState.foot_contact_bool.tolist())
        observation.extend(self._robot.currentState.body_linvel.tolist())
        observation.extend(self._robot.currentState.body_angvel.tolist())
        return observation


    def GetObservationContactsUpperBound(self):
        """Get the upper bound of the observation.

        Returns:
          The upper bound of an observation. See GetObservation() for the details
            of each element of an observation.
        """
        upper_bound = np.array([0.0] * len(self.GetObservationContacts()))
        #upper_bound[0:3*self.num_motors+4] = self.GetObservationUpperBound()
        upper_bound[0:self.num_motors] = (self._robot_config.UPPER_ANGLE_JOINT )#np.pi  # Joint angle.
        upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
        upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
        upper_bound[3*self.num_motors+4] = 1.0  # Body z .
        upper_bound[3*self.num_motors+5:3*self.num_motors+5+4] = [1.]*4  # In contact bools
        upper_bound[3*self.num_motors+9:3*self.num_motors+12] = (4.,4.,4.) # base lin vel max
        upper_bound[3*self.num_motors+12:] = (4.,4.,4.) # base ang vel max
        return upper_bound

    def GetObservationContactsLowerBound(self):
        """Get the lower bound of the observation.

        Returns:
          The lower bound of an observation. See GetObservation() for the details
            of each element of an observation.
        """
        lower_bound = np.array([0.0] * len(self.GetObservationContacts()))
        #lower_bound[0:3*self.num_motors+4] = -self.GetObservationUpperBound() #self.GetObservationLowerBound()
        lower_bound[0:self.num_motors] = (self._robot_config.LOWER_ANGLE_JOINT )#np.pi  # Joint angle.
        lower_bound[self.num_motors:2 * self.num_motors] = (-self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
        lower_bound[2 * self.num_motors:3 * self.num_motors] = (-self._robot_config.TORQUE_LIMITS)  # Joint torque.
        lower_bound[3*self.num_motors+4] = self._robot_config.IS_FALLEN_HEIGHT  # Body z .
        lower_bound[3*self.num_motors+5:3*self.num_motors+5+4] = [0.]*4  # In contact bools
        lower_bound[3*self.num_motors+9:3*self.num_motors+12] = (-4.,-4.,-4.) # base lin vel max
        lower_bound[3*self.num_motors+12:] = (-4.,-4.,-4.) # base ang vel max
        return lower_bound

    # def GetObservationUpperBound(self):
    #     """Get the upper bound of the observation.

    #     Returns:
    #       The upper bound of an observation. See GetObservation() for the details
    #         of each element of an observation.
    #     """
    #     upper_bound = np.array([0.0] * len(self.GetObservation()))
    #     upper_bound[0:self.num_motors] = (self._robot_config.UPPER_ANGLE_JOINT )#np.pi  # Joint angle.
    #     upper_bound[self.num_motors:2 * self.num_motors] = (self._robot_config.VELOCITY_LIMITS)  # Joint velocity.
    #     upper_bound[2 * self.num_motors:3 * self.num_motors] = (self._robot_config.TORQUE_LIMITS)  # Joint torque.
    #     upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
    #     return upper_bound