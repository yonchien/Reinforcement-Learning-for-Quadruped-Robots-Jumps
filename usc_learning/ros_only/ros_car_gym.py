""" ROS car gym, to be used with RL 

WORK IN PROGRESS
"""
import rospy
from geometry_msgs.msg import WrenchStamped, Vector3
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from control_msgs.msg import JointControllerState

from std_srvs.srv import Empty, EmptyRequest
from std_msgs.msg import Empty as EmptyMsg
from std_msgs.msg import Int32

import numpy as np
import time
from datetime import datetime

#from usc_learning.ros_only.gazebo_comm import GazeboComm


# gym
import gym
from gym import spaces

# RL setup
OBSERVATION_EPS = 0.01
rospy.init_node('ros_car_gym', anonymous=True)

"""
Task modes:
Output steering and torque/vel
TORQUE
VEL

Task envs:
PARKING - get to some particular xy theta
DRIFTING - learn to drift autonomously 

"""

class ROSCarGymEnv(gym.Env):
    """ Pure ROS gym env for a car """

    def __init__(self,
            #
            task_mode="TORQUE",
            task_env="PARKING",
            observation_space_mode="DEFAULT_CONTACTS",
            **kwargs
        ):
    self._task_mode = task_mode
    self._task_env = task_env

    self._setup_ros_comm()
    self.reset()
    self.setupObsAndActionSpaces()


    def _setup_ros_comm(self):
        """Setup and necessary ROS comm. """
        # reset MPC
        self._reset_env_pub = rospy.Publisher("/racecar_gazebo/reset_env", EmptyMsg, queue_size=1)
        # set gait
        self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)
        # msg containing v_des, yaw_rate_des, rlFootPos
        self._setRLActions_pub = rospy.Publisher("/laikago_gazebo/set_rl_foot_pos", RLFootPos, queue_size=1)

        # get car state
        self._base_link_sub = rospy.Subscriber("/base_link/odom",Vector3, self.baseLinkCallback, tcp_nodelay=True )
        self._vesc_odom_sub = rospy.Subscriber("/vesc/odom",Odometry, self.legCmdsCallback, tcp_nodelay=True )

        self._left_steering_sub = rospy.Subscriber("/racecar/left_steering_hinge_position_controller/state", JointControllerState, self.leftSteeringCallback, tcp_nodelay=True )
        self._right_steering_sub = rospy.Subscriber("/racecar/right_steering_hinge_position_controller/state", JointControllerState, self.rightSteeringCallback, tcp_nodelay=True )

        self.r_buffer = deque(maxlen=10)
        #r_buffer.push_back(0); //(1,0);

        rospy.wait_for_service('/laikago_gazebo/reset_env_service')
        try:
            self.reset_env_service = rospy.ServiceProxy('/laikago_gazebo/reset_env_service', Empty, persistent=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        # create msgs
        #self.rlFootPos = RLFootPos()


    def baseLinkCallback(self, msg):
        """ """
        self.vx = msg.x
        self.vy = msg.y

    def vescOdomCallback(self, msg):
        """ """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.thb = euler_from_quaternion(msg.pose.pose.orientation)[2] + np.pi/2
        self.r = msg.twist.twist.angular.z
        self.r_buffer.append(self.r)

    def leftSteeringCallback(self,msg):
        self.delta_l = msg.process_value

    def rightSteeringCallback(self,msg):
        self.delta_r = msg.process_value

    def get_state(self):
        """ Get car state """
        return np.array([self.x,
                         self.y,
                         self.thb,
                         self.vx,
                         self.vy,
                         np.mean(self.r_buffer),
                         0.5 * (self.delta_r - self.delta_l) ])



    ##########################################################################################
    # Observation
    ##########################################################################################

    def _get_observation(self):
        if self._observation_space_mode == "DEFAULT_CONTACTS":
            self._observation = np.array(self.GetObservationContacts())
            self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
            self.GetObservationContactsUpperBound())
        else:
            raise ValueError("observation space not defined or not intended")
        #self._observation = self._r

        # return additional items depending on mode (i.e. desired parking spot, or desired velocities for autnomous drifting)