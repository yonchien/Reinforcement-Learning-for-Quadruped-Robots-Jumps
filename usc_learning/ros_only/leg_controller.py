""" LegController.h/cpp """
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from laikago_msgs.msg import LowState, MotorState, LowCmd
from gazebo_msgs.msg import ModelStates

import numpy as np

from usc_learning.ros_only.current_state import lowState

"""
Notes:
Iterating through lowState and the leg objects is not optimal.
TODO: move everything to numpy
"""

lowCmd = LowCmd()

class LegControllerCommand():
    """ Same as cpp struct """
    def __init__(self):
        self.zero()

    def zero(self):
        # Vec3
        self.qDes = np.zeros(3)
        self.qdDes = np.zeros(3)
        self.tau = np.zeros(3)
        self.pDes = np.zeros(3)
        self.vDes = np.zeros(3)
        self.feedforwardForce = np.zeros(3)
        self.kpCartesian = np.zeros((3,3))
        self.kdCartesian = np.zeros((3,3))

class LegControllerData():
    """ Same as cpp struct """
    def __init__(self):
        self.zero()

    def zero(self):
        # Vec3
        self.q = np.zeros(3)
        self.qd = np.zeros(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.J = np.zeros((3,3))
        self.tau = np.zeros(3)

class LegController():
    def __init__(self, robot_config):
        self._quad = robot_config

        self.commands = [LegControllerCommand()] * 4
        self.data = [LegControllerData()] * 4

    def updateData(self):
        for leg in range(4):
            for j in range(3):
                self.data[leg].q[j] = lowState.motorState[leg*3+j].position
                self.data[leg].qd[j] = lowState.motorState[leg*3+j].velocity
                self.data[leg].tau[j] = lowState.motorState[leg*3+j].torque

            self.computeLegJacobianAndPosition(self.data[leg].q, leg)
            # v
            self.data[leg].v = self.data[leg].J @ self.data[leg].qd

    def updateCommand(self):
        """ NOTE: MUST call sendServoCmd() after to actually send commands """
        for i in range(4):
            J, pos = self.computeLegJacobianAndPosition(self.data[i].q,i)
            # tauFF
            legTorque = self.commands[i].tau
            # des ground force
            footForce = self.commands[i].feedforwardForce \
                        + self.commands[i].kpCartesian @ (self.commands[i].pDes - self.data[i].p) \
                        + self.commands[i].kdCartesian @ (self.commands[i].vDes - self.data[i].v)

            legTorque = self.data[i].J.T @ footForce
            self.commands[i].tau = legTorque

            for j in range(3):
                lowCmd.motorCmd[i*3+j].torque = self.commands[i].tau[j]


    def computeLegJacobianAndPosition(self, q, legID):
        """Translated to python/pybullet from: 
        https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/LegController.cpp
        can also use pybullet function calculateJacobian()
        IN LEG FRAME
        (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
        # from LegController:
        # * Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
        """
        pos = np.zeros(3)

        l1 = self._quad.HIP_LINK_LENGTH
        l2 = self._quad.THIGH_LINK_LENGTH
        l3 = self._quad.CALF_LINK_LENGTH

        sideSign = 1
        if legID == 0 or legID == 2:
          sideSign = -1

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        J = np.zeros((3,3))

        J[1, 0] = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
        J[2, 0] =  sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
        J[0, 1] = -l3 * c23 - l2 * c2
        J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
        J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
        J[0, 2] = -l3 * c23
        J[1, 2] = -l3 * s23 *s1
        J[2, 2] = l3 * s23 * c1  

        # E.E. pos
        pos[0] = -l3 * s23 - l2 * s2
        pos[1] = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        pos[2] = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

        return J, pos

