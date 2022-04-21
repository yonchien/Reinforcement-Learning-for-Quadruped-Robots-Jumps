""" This combines Quadruped.h/cpp and body.h/cpp, to be also used as gym? """
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from laikago_msgs.msg import LowState, MotorState, LowCmd, MotorCmd
from gazebo_msgs.msg import ModelStates

import numpy as np
import time
from datetime import datetime

from usc_learning.ros_only.current_state import CurrentState, lowState
from usc_learning.ros_only.leg_controller import LegController, lowCmd

#lowCmd = LowCmd()
#lowState = LowState()

"""
TODO:
add servo_pub


"""
# // needed to update commands
servo_pub = [0] * 12
servo_pub[0] = rospy.Publisher("/laikago_gazebo/FR_hip_controller/command", MotorCmd, queue_size=1)
servo_pub[1] = rospy.Publisher("/laikago_gazebo/FR_thigh_controller/command", MotorCmd, queue_size=1)
servo_pub[2] = rospy.Publisher("/laikago_gazebo/FR_calf_controller/command", MotorCmd, queue_size=1)
servo_pub[3] = rospy.Publisher("/laikago_gazebo/FL_hip_controller/command", MotorCmd, queue_size=1)
servo_pub[4] = rospy.Publisher("/laikago_gazebo/FL_thigh_controller/command", MotorCmd, queue_size=1)
servo_pub[5] = rospy.Publisher("/laikago_gazebo/FL_calf_controller/command", MotorCmd, queue_size=1)
servo_pub[6] = rospy.Publisher("/laikago_gazebo/RR_hip_controller/command", MotorCmd, queue_size=1)
servo_pub[7] = rospy.Publisher("/laikago_gazebo/RR_thigh_controller/command", MotorCmd, queue_size=1)
servo_pub[8] = rospy.Publisher("/laikago_gazebo/RR_calf_controller/command", MotorCmd, queue_size=1)
servo_pub[9] = rospy.Publisher("/laikago_gazebo/RL_hip_controller/command", MotorCmd, queue_size=1)
servo_pub[10] = rospy.Publisher("/laikago_gazebo/RL_thigh_controller/command", MotorCmd, queue_size=1)
servo_pub[11] = rospy.Publisher("/laikago_gazebo/RL_calf_controller/command", MotorCmd, queue_size=1)

class Body():
    def __init__(self):
        pass
    ##########################################################################################
    ##########################################################################################
    # Below is body.h/cpp
    ##########################################################################################
    ##########################################################################################

    ##########################################################################################
    # Functions for walking 
    ##########################################################################################
    def paramInit(self):

        for i in range(4):
            lowCmd.motorCmd[i*3+0].mode = 0x0A;
            lowCmd.motorCmd[i*3+0].positionStiffness = 0 #;//70;
            lowCmd.motorCmd[i*3+0].velocity = 0;
            lowCmd.motorCmd[i*3+0].velocityStiffness = 0 #;//3;
            lowCmd.motorCmd[i*3+0].torque = 0;
            lowCmd.motorCmd[i*3+1].mode = 0x0A;
            lowCmd.motorCmd[i*3+1].positionStiffness = 0 #;//180;
            lowCmd.motorCmd[i*3+1].velocity = 0;
            lowCmd.motorCmd[i*3+1].velocityStiffness = 0 #;//8;
            lowCmd.motorCmd[i*3+1].torque = 0;
            lowCmd.motorCmd[i*3+2].mode = 0x0A;
            lowCmd.motorCmd[i*3+2].positionStiffness = 0 #;//300;
            lowCmd.motorCmd[i*3+2].velocity = 0;
            lowCmd.motorCmd[i*3+2].velocityStiffness = 0 #;//15;
            lowCmd.motorCmd[i*3+2].torque = 0;
 
    def stand(self):
        pos = [ 0.0, 0.5, -1.4, -0.0, 0.5, -1.4, 
                0.0, 0.5, -1.4, -0.0, 0.5, -1.4]
        self.moveAllPosition(pos, 2000)

    def motion_init(self):
        self.paramInit()
        self.stand()

    def sendServoCmd(self):
        for m in range(12):
            servo_pub[m].publish(lowCmd.motorCmd[m])
    
        #ros::spinOnce();
        #//usleep(1000); // in microseconds

    def moveAllPosition(self, targetPos, duration):
        #double pos[12] ,lastPos[12], percent;
        pos = np.zeros(12)
        lastPos = np.zeros(12)
        percent = 0
        for j in range(12): 
            lastPos[j] = lowState.motorState[j].position
        for i in range(1,duration+1): #(int i=1; i<=duration; i++){
            if rospy.is_shutdown(): 
                break #(!ros::ok()) break;
            percent = i/duration
            for j in range(12):
                lowCmd.motorCmd[j].position = lastPos[j]*(1-percent) + targetPos[j]*percent

            self.sendServoCmd()
            time.sleep(500e-6) #usleep(500)

    def motion_reset(self):
        for i in range(4): 
            lowCmd.motorCmd[i*3+0].mode = 0x0A;
            lowCmd.motorCmd[i*3+0].positionStiffness = 70;
            lowCmd.motorCmd[i*3+0].velocity = 0;
            lowCmd.motorCmd[i*3+0].velocityStiffness = 3;
            lowCmd.motorCmd[i*3+0].torque = 0;
            lowCmd.motorCmd[i*3+1].mode = 0x0A;
            lowCmd.motorCmd[i*3+1].positionStiffness = 180;
            lowCmd.motorCmd[i*3+1].velocity = 0;
            lowCmd.motorCmd[i*3+1].velocityStiffness = 8;
            lowCmd.motorCmd[i*3+1].torque = 0;
            lowCmd.motorCmd[i*3+2].mode = 0x0A;
            lowCmd.motorCmd[i*3+2].positionStiffness = 300;
            lowCmd.motorCmd[i*3+2].velocity = 0;
            lowCmd.motorCmd[i*3+2].velocityStiffness = 15;
            lowCmd.motorCmd[i*3+2].torque = 0;

        pos = [-0.0, 0.5, -1.4, -0.0, 0.5, -1.4, 
                0.0, 0.5, -1.4, -0.0, 0.5, -1.4];
        self.moveAllPosition(pos, 5000)
        self.sendServoCmd()

    ##########################################################################################
    # Functions for jumping  
    ##########################################################################################

    def paramJumpInit(self):
        for i in range(4):
            lowCmd.motorCmd[i*3+0].mode = 0x0A;
            lowCmd.motorCmd[i*3+0].positionStiffness = 70;
            lowCmd.motorCmd[i*3+0].velocity = 0;
            lowCmd.motorCmd[i*3+0].velocityStiffness = 3;
            lowCmd.motorCmd[i*3+0].torque = 0;
            lowCmd.motorCmd[i*3+1].mode = 0x0A;
            lowCmd.motorCmd[i*3+1].positionStiffness = 180;
            lowCmd.motorCmd[i*3+1].velocity = 0;
            lowCmd.motorCmd[i*3+1].velocityStiffness = 8;
            lowCmd.motorCmd[i*3+1].torque = 0;
            lowCmd.motorCmd[i*3+2].mode = 0x0A;
            lowCmd.motorCmd[i*3+2].positionStiffness = 300;
            lowCmd.motorCmd[i*3+2].velocity = 0;
            lowCmd.motorCmd[i*3+2].velocityStiffness = 15;
            lowCmd.motorCmd[i*3+2].torque = 0;

    def JumpPrepare(self):
        targetPos = [0, 1.2566, -2.355, 0, 1.2566, -2.355, 
                     0, 1.2566, -2.355, 0, 1.2566, -2.355]
        lastPos = np.zeros(12)
        percent = 0
        duration = 1500
        for j in range(12):
            lastPos[j] = lowState.motorState[j].position
        for i in range(1,duration+1):
            if rospy.is_shutdown(): 
                break
            percent = i/duration
            for j in range(12):
                lowCmd.motorCmd[j].position = lastPos[j]*(1-percent) + targetPos[j]*percent
            
            self.sendPrepareCmd();

    def Jump_Init(self):
        self.paramJumpInit();
        self.JumpPrepare();

    def sendPrepareCmd(self):
        for m in range(12):
            servo_pub[m].publish(lowCmd.motorCmd[m])
        #ros::spinOnce();
        time.sleep(1000e-6) #usleep(1000); // in microseconds


class Quadruped():
    def __init__(self, robot_config):

        self.body = Body()
        self.legController = LegController(robot_config)
        self.currentState = CurrentState()

    def sendServoCmd(self):
        self.legController.updateCommand()
        self.body.sendServoCmd()

    def reset(self):
        self.body.motion_reset()

    def reset_jump(self):
        self.body.Jump_Init()


def jumping():
    # load in trajectory, execute on robot
    from usc_learning.imitation_tasks.traj_motion_data import TrajTaskMotionData
    import usc_learning.envs.quadruped_master.configs_a1 as robot_config
    opt_data = 'data2/jumpingFull_A1_1ms_h50_d60.csv'
    DO_IMPEDANCE = False
    USE_DEFAULT_DATA = False
    if not DO_IMPEDANCE:
        traj_task = TrajTaskMotionData(filename=opt_data,dt=0.001)
    else:
        traj_task = TrajTaskMotionData(filename=opt_data,dt=0.001,useCartesianData=True)

    # ros set up
    ros_rate = rospy.Rate(200)
    # set up quadruped object
    quadruped = Quadruped(robot_config)
    time.sleep(2)
    quadruped.body.Jump_Init()
    time.sleep(2)
    Kp = 200
    Kd = 50
    Hip_Kp = 300
    Hip_Kd = 100

    KP = np.array([Hip_Kp, Kp, Kp] * 4)
    KD = np.array([Hip_Kd, Kd, Kd] * 4)
    print('Start jumping')

    trajLen = traj_task.trajLen
    ################################################# check existing data
    if USE_DEFAULT_DATA:
        # TODO: remove absolute paths
        data_q = '/home/guillaume/catkin_ws/src/USC-AlienGo-Software/laikago_ros/optimization_data/data_Q.csv'
        data_tau = '/home/guillaume/catkin_ws/src/USC-AlienGo-Software/laikago_ros/optimization_data/data_tau.csv'

        traj_Q = np.loadtxt(data_q, delimiter=',')
        traj_tau = np.loadtxt(data_tau, delimiter=',')

        trajLen = traj_Q.shape[1]
        q  = traj_Q[3:7,:]
        dq = traj_Q[10:14,:]
        torques = traj_tau
        # map from 2D optimization to 3D aliengo in pybullet
        qFront = -q[0:2,:]
        qRear  = -q[2:4,:]
        dqFront = -dq[0:2,:]
        dqRear  = -dq[2:4,:]
        torquesFront = -0.5*torques[0:2,:]
        torquesRear  = -0.5*torques[2:4,:]
        # fill in full q, dq, torque arrays
        frontIndices = np.array([1,2,4,5])
        rearIndices  = np.array([7,8,10,11])
        q = np.zeros((12,trajLen))
        dq = np.zeros((12,trajLen))
        taus = np.zeros((12,trajLen))

        q[frontIndices,:] = np.vstack((qFront,qFront))
        q[rearIndices,:]  = np.vstack((qRear,qRear))
        dq[frontIndices,:] = np.vstack((dqFront,dqFront))
        dq[rearIndices,:]  = np.vstack((dqRear,dqRear))
        taus[frontIndices,:] = np.vstack((torquesFront,torquesFront))
        taus[rearIndices,:]  = np.vstack((torquesRear,torquesRear))

    #################################################

    if not rospy.is_shutdown():

        for i in range(0,trajLen,5):

            temptime = datetime.now() 
            quadruped.legController.updateData()

            # des_state = traj_task.get_state_at_index(i)
            # des_torques = -1*traj_task.get_torques_at_index(i)
            # q_des = traj_task.get_joint_pos_from_state(des_state)
            # dq_des = traj_task.get_joint_vel_from_state(des_state)
            if USE_DEFAULT_DATA:
                des_torques = taus[:,i]
                q_des = q[:,i]
                dq_des = dq[:,i]
            else:
                des_state = traj_task.get_state_at_index(i)
                des_torques = -1*traj_task.get_torques_at_index(i)
                q_des = traj_task.get_joint_pos_from_state(des_state)
                dq_des = traj_task.get_joint_vel_from_state(des_state)

            for j in range(12):
                lowCmd.motorCmd[j].position = q_des[j]
                lowCmd.motorCmd[j].velocity = dq_des[j]
                lowCmd.motorCmd[j].torque = des_torques[j]
                lowCmd.motorCmd[j].positionStiffness = KP[j]
                lowCmd.motorCmd[j].velocityStiffness = KD[j]

            quadruped.sendServoCmd()

            #
            ros_rate.sleep()
            print(datetime.now() - temptime)

    quadruped.body.Jump_Init()


if __name__ == '__main__':
    # init 
    rospy.init_node('jumping_test', anonymous=True)
    jumping()
    # ros_rate = rospy.Rate(10)
    # # setup
    # current_state = CurrentState()

    # while not rospy.is_shutdown():
    #     print(lowState)
    #     ros_rate.sleep()