""" CurrentState.h/cpp """
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Imu
from laikago_msgs.msg import LowState, MotorState
from gazebo_msgs.msg import ModelStates

import numpy as np

"""
Notes:
Keeping everything in lowState and then having to iterate through each time is not optimal.
TODO: move everything to numpy
"""

lowState = LowState()

class CurrentState():
    """ Class mirroring CurrentState.h/cpp """   
    def __init__(self):

        # keep relevant info in numpy arrays
        self.body_pos = np.zeros(3)
        self.body_orn = np.zeros(4)
        self.body_linvel = np.zeros(3)
        self.body_angvel = np.zeros(3)
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.joint_tau = np.zeros(12)
        self.foot_contact_bool = np.zeros(4)
        
        self.imu_sub = rospy.Subscriber("/trunk_imu", Imu, self.imuCallback, queue_size=1)
        self.state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.cheaterCallback, queue_size=1) 

        self.footForce_sub_0 = rospy.Subscriber("/visual/FR_foot_contact/the_force", WrenchStamped, self.FRfootCallback, queue_size=1)
        self.footForce_sub_1 = rospy.Subscriber("/visual/FL_foot_contact/the_force", WrenchStamped, self.FLfootCallback, queue_size=1)
        self.footForce_sub_2 = rospy.Subscriber("/visual/RR_foot_contact/the_force", WrenchStamped, self.RRfootCallback, queue_size=1)
        self.footForce_sub_3 = rospy.Subscriber("/visual/RL_foot_contact/the_force", WrenchStamped, self.RLfootCallback, queue_size=1)
        self.servo_sub_0 = rospy.Subscriber("/laikago_gazebo/FR_hip_controller/state", MotorState, self.FRhipCallback, queue_size=1)
        self.servo_sub_1 = rospy.Subscriber("/laikago_gazebo/FR_thigh_controller/state", MotorState, self.FRthighCallback, queue_size=1)
        self.servo_sub_2 = rospy.Subscriber("/laikago_gazebo/FR_calf_controller/state", MotorState, self.FRcalfCallback, queue_size=1)
        self.servo_sub_3 = rospy.Subscriber("/laikago_gazebo/FL_hip_controller/state", MotorState, self.FLhipCallback, queue_size=1) 
        self.servo_sub_4 = rospy.Subscriber("/laikago_gazebo/FL_thigh_controller/state", MotorState, self.FLthighCallback, queue_size=1)
        self.servo_sub_5 = rospy.Subscriber("/laikago_gazebo/FL_calf_controller/state", MotorState, self.FLcalfCallback, queue_size=1)
        self.servo_sub_6 = rospy.Subscriber("/laikago_gazebo/RR_hip_controller/state", MotorState, self.RRhipCallback, queue_size=1)
        self.servo_sub_7 = rospy.Subscriber("/laikago_gazebo/RR_thigh_controller/state", MotorState, self.RRthighCallback, queue_size=1)
        self.servo_sub_8 = rospy.Subscriber("/laikago_gazebo/RR_calf_controller/state", MotorState, self.RRcalfCallback, queue_size=1)
        self.servo_sub_9 = rospy.Subscriber("/laikago_gazebo/RL_hip_controller/state", MotorState, self.RLhipCallback, queue_size=1)
        self.servo_sub_10 = rospy.Subscriber("/laikago_gazebo/RL_thigh_controller/state", MotorState, self.RLthighCallback, queue_size=1)
        self.servo_sub_11 = rospy.Subscriber("/laikago_gazebo/RL_calf_controller/state", MotorState, self.RLcalfCallback, queue_size=1)


    ######################################

    def cheaterCallback(self, msg):
        """ Gazebo ModelStates msg """
        lowState.cheat.orientation[0] = msg.pose[2].orientation.w;
        lowState.cheat.orientation[1] = msg.pose[2].orientation.x;
        lowState.cheat.orientation[2] = msg.pose[2].orientation.y;
        lowState.cheat.orientation[3] = msg.pose[2].orientation.z;

        lowState.cheat.position[0] = msg.pose[2].position.x;
        lowState.cheat.position[1] = msg.pose[2].position.y;
        lowState.cheat.position[2] = msg.pose[2].position.z;

        lowState.cheat.vBody[0] = msg.twist[2].linear.x;
        lowState.cheat.vBody[1] = msg.twist[2].linear.y;
        lowState.cheat.vBody[2] = msg.twist[2].linear.z;

        lowState.cheat.omegaBody[0] = msg.twist[2].angular.x;
        lowState.cheat.omegaBody[1] = msg.twist[2].angular.y;
        lowState.cheat.omegaBody[2] = msg.twist[2].angular.z;

        self.body_pos = np.array([msg.pose[2].position.x, msg.pose[2].position.y, msg.pose[2].position.z])
        self.body_orn = np.array([msg.pose[2].orientation.w, msg.pose[2].orientation.x, msg.pose[2].orientation.y, msg.pose[2].orientation.z])
        self.body_linvel = np.array([msg.twist[2].linear.x, msg.twist[2].linear.y, msg.twist[2].linear.z])
        self.body_angvel = np.array([msg.twist[2].angular.x, msg.twist[2].angular.y, msg.twist[2].angular.z])
    

    def imuCallback(self, msg):
        """Gazebo sensor_msgs Imu  """
        #print('in imuCallback, msg:', msg)
        lowState.imu.quaternion[3] = msg.orientation.w;
        lowState.imu.quaternion[0] = msg.orientation.x;
        lowState.imu.quaternion[1] = msg.orientation.y;
        lowState.imu.quaternion[2] = msg.orientation.z;

        lowState.imu.acceleration[0] = msg.linear_acceleration.x;
        lowState.imu.acceleration[1] = msg.linear_acceleration.y;
        lowState.imu.acceleration[2] = msg.linear_acceleration.z;

        lowState.imu.gyroscope[0] = msg.angular_velocity.x;
        lowState.imu.gyroscope[1] = msg.angular_velocity.y;
        lowState.imu.gyroscope[2] = msg.angular_velocity.z;
        #print(lowState)

    ###################################### FR
    def FRhipCallback(self, msg):
        #//start_up = false;
        lowState.motorState[0].mode = msg.mode;
        lowState.motorState[0].position = msg.position;
        lowState.motorState[0].velocity = msg.velocity;
        lowState.motorState[0].torque = msg.torque;

        self.joint_pos[0] = msg.position
        self.joint_vel[0] = msg.velocity
        self.joint_tau[0] = msg.torque

    def FRthighCallback(self, msg):
        lowState.motorState[1].mode = msg.mode;
        lowState.motorState[1].position = msg.position;
        lowState.motorState[1].velocity = msg.velocity;
        lowState.motorState[1].torque = msg.torque;

        self.joint_pos[1] = msg.position
        self.joint_vel[1] = msg.velocity
        self.joint_tau[1] = msg.torque

    def FRcalfCallback(self, msg):
        lowState.motorState[2].mode = msg.mode;
        lowState.motorState[2].position = msg.position;
        lowState.motorState[2].velocity = msg.velocity;
        lowState.motorState[2].torque = msg.torque;

        self.joint_pos[2] = msg.position
        self.joint_vel[2] = msg.velocity
        self.joint_tau[2] = msg.torque

    ###################################### FL
    def FLhipCallback(self, msg):
        lowState.motorState[3].mode = msg.mode;
        lowState.motorState[3].position = msg.position;
        lowState.motorState[3].velocity = msg.velocity;
        lowState.motorState[3].torque = msg.torque;

        self.joint_pos[3] = msg.position
        self.joint_vel[3] = msg.velocity
        self.joint_tau[3] = msg.torque

    def FLthighCallback(self, msg):
        lowState.motorState[4].mode = msg.mode;
        lowState.motorState[4].position = msg.position;
        lowState.motorState[4].velocity = msg.velocity;
        lowState.motorState[4].torque = msg.torque;

        self.joint_pos[4] = msg.position
        self.joint_vel[4] = msg.velocity
        self.joint_tau[4] = msg.torque

    def FLcalfCallback(self, msg):
        lowState.motorState[5].mode = msg.mode;
        lowState.motorState[5].position = msg.position;
        lowState.motorState[5].velocity = msg.velocity;
        lowState.motorState[5].torque = msg.torque;

        self.joint_pos[5] = msg.position
        self.joint_vel[5] = msg.velocity
        self.joint_tau[5] = msg.torque

    ###################################### RR
    def RRhipCallback(self, msg):
        lowState.motorState[6].mode = msg.mode;
        lowState.motorState[6].position = msg.position;
        lowState.motorState[6].velocity = msg.velocity;
        lowState.motorState[6].torque = msg.torque;

        self.joint_pos[6] = msg.position
        self.joint_vel[6] = msg.velocity
        self.joint_tau[6] = msg.torque

    def RRthighCallback(self, msg):
        lowState.motorState[7].mode = msg.mode;
        lowState.motorState[7].position = msg.position;
        lowState.motorState[7].velocity = msg.velocity;
        lowState.motorState[7].torque = msg.torque;

        self.joint_pos[7] = msg.position
        self.joint_vel[7] = msg.velocity
        self.joint_tau[7] = msg.torque

    def RRcalfCallback(self, msg): 
        lowState.motorState[8].mode = msg.mode;
        lowState.motorState[8].position = msg.position;
        lowState.motorState[8].velocity = msg.velocity;
        lowState.motorState[8].torque = msg.torque;

        self.joint_pos[8] = msg.position
        self.joint_vel[8] = msg.velocity
        self.joint_tau[8] = msg.torque

    ###################################### RL
    def RLhipCallback(self, msg):
        #//start_up = false;
        lowState.motorState[9].mode = msg.mode;
        lowState.motorState[9].position = msg.position;
        lowState.motorState[9].velocity = msg.velocity;
        lowState.motorState[9].torque = msg.torque;

        self.joint_pos[9] = msg.position
        self.joint_vel[9] = msg.velocity
        self.joint_tau[9] = msg.torque   

    def RLthighCallback(self, msg):
        lowState.motorState[10].mode = msg.mode;
        lowState.motorState[10].position = msg.position;
        lowState.motorState[10].velocity = msg.velocity;
        lowState.motorState[10].torque = msg.torque;

        self.joint_pos[10] = msg.position
        self.joint_vel[10] = msg.velocity
        self.joint_tau[10] = msg.torque

    def RLcalfCallback(self, msg):
        lowState.motorState[11].mode = msg.mode;
        lowState.motorState[11].position = msg.position;
        lowState.motorState[11].velocity = msg.velocity;
        lowState.motorState[11].torque = msg.torque;

        self.joint_pos[11] = msg.position
        self.joint_vel[11] = msg.velocity
        self.joint_tau[11] = msg.torque

    ###################################### foot callbacks
    def FRfootCallback(self, msg):
        lowState.eeForce[0].x = msg.wrench.force.x;
        lowState.eeForce[0].y = msg.wrench.force.y;
        lowState.eeForce[0].z = msg.wrench.force.z;
        lowState.footForce[0] = msg.wrench.force.z;

        self.foot_contact_bool[0] = 1 if msg.wrench.force.z > 0 else 0

    def FLfootCallback(self, msg):
        lowState.eeForce[1].x = msg.wrench.force.x;
        lowState.eeForce[1].y = msg.wrench.force.y;
        lowState.eeForce[1].z = msg.wrench.force.z;
        lowState.footForce[1] = msg.wrench.force.z;

        self.foot_contact_bool[1] = 1 if msg.wrench.force.z > 0 else 0

    def RRfootCallback(self, msg):
        lowState.eeForce[2].x = msg.wrench.force.x;
        lowState.eeForce[2].y = msg.wrench.force.y;
        lowState.eeForce[2].z = msg.wrench.force.z;
        lowState.footForce[2] = msg.wrench.force.z;

        self.foot_contact_bool[2] = 1 if msg.wrench.force.z > 0 else 0

    def RLfootCallback(self, msg):
        lowState.eeForce[3].x = msg.wrench.force.x;
        lowState.eeForce[3].y = msg.wrench.force.y;
        lowState.eeForce[3].z = msg.wrench.force.z;
        lowState.footForce[3] = msg.wrench.force.z;

        self.foot_contact_bool[3] = 1 if msg.wrench.force.z > 0 else 0



if __name__ == '__main__':
    # init 
    rospy.init_node('current_state_test', anonymous=True)
    ros_rate = rospy.Rate(10)
    # setup
    current_state = CurrentState()

    while not rospy.is_shutdown():
        print(lowState)
        ros_rate.sleep()