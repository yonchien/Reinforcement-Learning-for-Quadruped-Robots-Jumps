from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
import usc_learning.envs.quadruped_master.configs_aliengo as robot_config


import os



# path_to_ros_source = "/home/shashank/Desktop/Research/DRCL/paper_fin/Imp_RL2/catkin_ws/devel/setup.bash"
# os.system("source " + path_to_ros_source)
#os.system("bash sourcer.sh")

from laikago_msgs.srv import MPCService, MPCServiceRequest, MPCServiceResponse
from laikago_msgs.msg import PybulletState, PybulletFullState

import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Empty, Int32
import numpy as np

rospy.init_node('pyb_interface', anonymous=True)

class quad_mpc():
    def __init__(self ):
        # service client to call MPC with pyb state, and return leg commands
        
        self._rate = rospy.Rate(1000) # careful
        rospy.wait_for_service('/laikago_gazebo/mpc_comm')
        try:
            self.mpc_comm = rospy.ServiceProxy('/laikago_gazebo/mpc_comm', MPCService,persistent=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        # publishers - reset MPC, and set gait
        self._resetMPC_pub = rospy.Publisher("/laikago_gazebo/reset_mpc", Empty, queue_size=1)
        self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)
        data = Int32()
        data.data = int(2)
        self._setGait_pub.publish(data)

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

    def resetMPC(self):
        """Reset MPC """
        self._resetMPC_pub.publish(Empty())

    def getAndSetActionFromMPC(self,info):
        # pyb_state = info

        data = PybulletFullState()

        data.position = info['position']
        data.orientation = info['orientation']
        data.rpy = info['rpy']
        data.rBody = info['rBody'] # numpy row indexed, eigen column indexed
        data.vBody = info['vBody']
        data.omegaBody = info['omegaBody']
        data.vWorld = info['vWorld']
        data.omegaWorld = info['omegaWorld']
        data.v_des = info['v_des']
        data.yaw_rate_des = info['yaw_rate_des']
        # set joint states
        for i in range(12):
            data.motorState[i].position = info['motorState'][i]['position']
            data.motorState[i].velocity = info['motorState'][i]['velocity']
            data.motorState[i].torque = info['motorState'][i]['torque']

        #print(data)
        leg_cmds = self.mpc_comm(data)
        return self.stepFromMPC(leg_cmds.legCmdArray, info)

    def stepFromMPC(self, data, info, calcTorque=True):
        tau_actions = np.array(data.motorCmd)
        all_Fns = np.zeros(12)
        all_pDes = np.zeros(12)
        for _ in range(1):
            if calcTorque:
                actions = np.zeros(12)
                for i in range(4):
                    kpCartesian = np.diag(data.legCmds[i].kpCartesianDiag)
                    kdCartesian = np.diag(data.legCmds[i].kdCartesianDiag)
                    u_force = data.legCmds[i].feedforwardForce
                    pDes = data.legCmds[i].pDes
                    vDes = data.legCmds[i].vDes
                    all_Fns[i*3:i*3+3] = u_force
                    all_pDes[i*3:i*3+3] = pDes
                    actions[i*3:i*3+3] = self._ComputeLegImpedanceControlActions(pDes,
                                                                                vDes,
                                                                                kpCartesian,
                                                                                kdCartesian,
                                                                                u_force,
                                                                                i, #legID,
                                                                                info)
                return actions
            else:
                return tau_actions  

    def _ComputeLegImpedanceControlActions(self,u_pos,u_vel,kpCartesian,kdCartesian,u_force,legID,info):
        joint_pos = [0] * 12
        joint_vel = [0] * 12
        joint_torques = [0] * 12

        for i in range(12):
            joint_pos[i] = info['motorState'][i]['position']
            joint_vel[i] = info['motorState'][i]['velocity']
            joint_torques[i] = info['motorState'][i]['torque']

        eepos = info['EEpos']
        #print(joint_pos)
        #print(eepos)

        J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(legID, joint_pos, eepos)
        Pd = u_pos 
        vd = u_vel 
        Fd = u_force 
            
        foot_linvel = J @ joint_vel[legID*3:legID*3+3]
        tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
        return tau

    def _ComputeJacobianAndPositionUSC(self, legID, joint_pos, eepos):
        # q = self.GetMotorAngles()[legID*3:legID*3+3]
        q = joint_pos[legID*3:legID*3+3]
        # linkPosInBase = self.GetLocalEEPos(legID)
        linkPosInBase = eepos[legID]
        pos = np.zeros(3)
        l1 = robot_config.HIP_LINK_LENGTH
        l2 = robot_config.THIGH_LINK_LENGTH
        l3 = robot_config.CALF_LINK_LENGTH
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
        pos[0] = -l3 * s23 - l2 * s2
        pos[1] = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        pos[2] = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2
        return J, pos


if __name__ == "__main__":

    env_config = {
        "isRLGymInterface": False, 
        "robot_config": robot_config, 
        "time_step": 0.001, # make sure this is small enough
        "render": True, 
        "motor_control_mode": "TORQUE"
    }
    env = QuadrupedGymEnv(**env_config)
    mpc = quad_mpc()

    state = env.reset()

    action = env.action_space.sample()

    for i in range(3000):
        state, reward, done, info = env.step(action)

        action = mpc.getAndSetActionFromMPC(info)

        if done:
            state = env.reset()
            mpc.resetMPC()