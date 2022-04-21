"""Example file, self-contained.

Receive data from ROS and visualize robot state in pybullet.

Need 3 terminals, with in order:

Terminal 1 (after source ROS path):
$ roslaunch laikago_gazebo aliengo.launch
Terminal 2 (after source ROS path):
$ rosrun laikago_gazebo laikago_pyb_send_state
Terminal 3 [this one] (source ROS path, activate virtualenv):
$ python example_vis_ros_state.py

"""
import sys
import time
import numpy as np
# ros imports
import rospy
from rospy.numpy_msg import numpy_msg
# get relevant msg formats
from laikago_msgs.msg import LowState, LegCmd, LegCmdArray, PybulletState

# pybullet env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# get appropriate config file
import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
#import usc_learning.envs.quadruped_master.configs_aliengo as robot_config

""" Example msgs being advertised:
"/laikago_gazebo/highState/state"
"/laikago_gazebo/lowState/state"
"/laikago_gazebo/FR_thigh_controller/command"
"/laikago_gazebo/legCmds"

Example msgs: PybulletData, PybulletDataArray, MotorCmd, HighState

-Note: ensure these are being published from gazebo sim
"""
rospy.init_node('mpc_listener', anonymous=True)

FR_foot = []
FL_foot = []
RR_foot = []
RL_foot = []
####################################################
# test listener
####################################################
class ROSDataListener(object):
    """Name something to make it clear this is for extracting the full robot state 
    """
    def __init__(self):
        """Setup any relevant data """
        # set up rospy listeners outside class
        # set states to none?
        self.env = QuadrupedGymEnv(isRLGymInterface=False, 
                        robot_config=robot_config, 
                        time_step=0.001, # make sure this is small enough
                        render=True, 
                        motor_control_mode="PD")
        self.hasStateData = False
        self.hasLegCmdData = False
        self.hasROSStateData = False
        self._lowState_sub = rospy.Subscriber("/laikago_gazebo/lowState/state",LowState, self.lowStateCallback )
        self._legCmds_sub = rospy.Subscriber("/laikago_gazebo/legCmds",LegCmdArray, self.legCmdsCallback )
        #self._rosState = rospy.Subscriber("/laikago_gazebo/ros2pybstate", PybulletState, self.getROSState)
        self._rate = rospy.Rate(1000) # careful

    def lowStateCallback(self, data):
        """See LowState.msg for format. 
        Can also print(data)

        TODO: check body vels
        """
        self.base_pos = data.cheat.position
        self.base_orn = np.array(data.cheat.orientation)
        # quaternion is in wrong order
        self.base_orn = np.roll(self.base_orn,-1)

        # check body vs world 
        self.base_linvel = data.cheat.vBody
        self.base_angvel = data.cheat.omegaBody
        joint_pos = np.zeros(12)
        joint_vel = np.zeros(12)
        joint_torques  = np.zeros(12)
        for i in range(12):
            joint_pos[i] = data.motorState[i].position
            joint_vel[i] = data.motorState[i].velocity
            joint_torques[i] = data.motorState[i].torque
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.joint_torques = joint_torques

        #print(data.foo)
        #print(data)
        FR_foot.append(data.footForce[0])
        FL_foot.append(data.footForce[1])
        RR_foot.append(data.footForce[2])
        RL_foot.append(data.footForce[3])
        self.hasStateData = True
        #print(self.base_pos)
        #print('base orn', self.base_orn)
        # if np.linalg.norm(self.base_orn - np.array([0,0,0,1])) > 0.05:
        #     sys.exit()

    def legCmdsCallback(self,data):
        """See LegCmd.msg and LegCmdArray.msg 
        Can access the data at run time when needed
        """
        #print('leg cmd data',data)
        self.legCmds = data
        self.hasLegCmdData = True

    def getROSState(self,data):
        """Get equivalent state form ROS, to compare
        """
        #data = PybulletState()
        # position
        pos = self.env._robot.GetBasePosition()
        # orientation
        orn = self.env._robot.GetBaseOrientation()
        orn =  np.roll(orn,1) #[orn[3], orn[0], orn[1], orn[2]]
        # rpy, rBody
        rpy = self.env._robot.GetBaseOrientationRollPitchYaw()
        R = self.env._robot.GetBaseOrientationMatrix()
        # world lin/ang vel
        vWorld = self.env._robot.GetBaseLinearVelocity()
        omegaWorld = self.env._robot.GetBaseAngularVelocity()
        # body lin/ang vel
        vBody = R @ vWorld
        omegaBody = self.env._robot.GetTrueBaseRollPitchYawRate()

        self.base_pos = data.position
        self.base_orn = np.roll(data.orientation,-1)

        print('*'*80, '\n COMPARING ROS WITH CURRENT PYB, should be close-ish')
        print('ros pos',data.position)
        print('pyb pos',pos)
        print('ros orn', data.orientation)
        print('pyb or', orn)
        print('ros rpy',data.rpy)
        print('pyb rpy', rpy)
        print('ros rBody',data.rBody)
        print('pyb rBody', R.ravel())
        print('ros vBody',data.vBody)
        print('pyb vBody', vBody)
        print('ros omegaBody',data.omegaBody)
        print('pyb omegaBody',omegaBody)
        print('ros vWorld',data.vWorld)
        print('pyb vWorld', vWorld)
        print('ros omegaWorld',data.omegaWorld)
        print('pyb omegaWorld',omegaWorld)
        #self._rate.sleep()
        self.real_base_linvel = data.vWorld
        self.real_base_angvel = data.omegaWorld
        self.hasROSStateData = True
####################################################
# Setup ros node, listener
####################################################
_rosBridge = ROSDataListener()

# wait a little
time.sleep(1)
counter = 0

#while not rospy.is_shutdown():
for dummy in range(200):
    if _rosBridge.hasStateData:# and _rosBridge.hasROSStateData:
        _rosBridge.env._robot.SetFullState(_rosBridge.base_pos,
                                _rosBridge.base_orn,
                                bodyLinVel=_rosBridge.base_linvel,
                                bodyAngVel=_rosBridge.base_angvel,
                                jointPositions=_rosBridge.joint_pos,
                                jointVels=_rosBridge.joint_vel)

        #if counter < 20:
        # print('*'*80, '\n counter', counter)
        # print('pos ros',_rosBridge.base_pos)
        # print('pos pyb', env._robot.GetBasePosition())
        # print('orn ros', _rosBridge.base_orn)
        # print('orn pyb', env._robot.GetBaseOrientation())
        # print('linvel ros',_rosBridge.base_linvel)
        # print('linvel pyb', env._robot.GetBaseLinearVelocity())
        # print('angvel ros',_rosBridge.base_angvel)
        # print('linvel pyb', env._robot.GetBaseAngularVelocity())
        # print('joint pos ros',_rosBridge.joint_pos)
        # print('joint pos pyb', env._robot.GetMotorAngles())
        # print('joint vel ros',_rosBridge.joint_vel)
        # print('joint vel pyb', env._robot.GetMotorVelocities())
        # counter += 1
        # for _ in range(100):
        #     _rosBridge.env._pybullet_client.stepSimulation()
            #_rosBridge.env._render_step_helper()

    #time.sleep(0.1)
    #_rosBridge._rate.sleep()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# plot ee forces    
plt.plot(FR_foot, label='FR')
plt.plot(FL_foot, label='FL')
plt.plot(RR_foot, label='RR')
plt.plot(RL_foot, label='RL')
plt.legend()
plt.show()