"""Example file, self-contained.

Send data to ROS, run MPC and get commands, execute in pybullet.

Need 3 terminals, with in order:

Terminal 1 (after source ROS path):
$ roslaunch laikago_gazebo aliengo.launch
Terminal 2 (after source ROS path):
$ rosrun laikago_gazebo laikago_pyb_mpc
Terminal 3 [this one] (source ROS path, activate virtualenv):
$ python example_send_state_to_ros.py

pyb_send_state.cc       # send ROS states/commands to pybullet (this one)
pyb_mpc_interface.cc    # receive robot state from pybullet, run MPC, send result back
"""
import time
import numpy as np
# ros imports
import rospy
from rospy.numpy_msg import numpy_msg
# get relevant msg formats
from laikago_msgs.msg import PybulletData, PybulletDataArray, MotorCmd, LowState, HighState, LegCmd, LegCmdArray, PybulletState, MotorState
from gazebo_msgs.msg import ModelStates, LinkStates, ModelState
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest
from std_srvs.srv import Empty, EmptyRequest

# pybullet env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# get appropriate config file
import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
#import usc_learning.envs.quadruped_master.configs_a1 as robot_config

""" Example msgs:
/laikago_gazebo/highState/state
/laikago_gazebo/lowState/state
/laikago_gazebo/FR_thigh_controller/command"
"/laikago_gazebo/legCmds"

-Note: ensure these are being published from gazebo sim
"""
rospy.init_node('pyb_interface', anonymous=True)
#####################################################
# joint names
#  - laikago_gazebo::base
gazebo_joint_names = [
    "laikago_gazebo::FL_hip",
    "laikago_gazebo::FL_thigh",
    "laikago_gazebo::FL_calf",
    "laikago_gazebo::FR_hip",
    "laikago_gazebo::FR_thigh",
    "laikago_gazebo::FR_calf",
    "laikago_gazebo::RL_hip",
    "laikago_gazebo::RL_thigh",
    "laikago_gazebo::RL_calf",
    "laikago_gazebo::RR_hip",
    "laikago_gazebo::RR_thigh",
    "laikago_gazebo::RR_calf",
]

motor_state_strings = [
    # FR
    "/laikago_gazebo/FR_hip_controller/state",
    "/laikago_gazebo/FR_thigh_controller/state",
    "/laikago_gazebo/FR_calf_controller/state",
    # FL
    "/laikago_gazebo/FL_hip_controller/state",
    "/laikago_gazebo/FL_thigh_controller/state",
    "/laikago_gazebo/FL_calf_controller/state",
    # RR
    "/laikago_gazebo/RR_hip_controller/state",
    "/laikago_gazebo/RR_thigh_controller/state",
    "/laikago_gazebo/RR_calf_controller/state",
    # RL
    "/laikago_gazebo/RL_hip_controller/state",
    "/laikago_gazebo/RL_thigh_controller/state",
    "/laikago_gazebo/RL_calf_controller/state"
]

####################################################
# test MPC Actor - just query ROS MPC from pybullet state
# don't worry about extracting action from RL yet, but soon
####################################################
class QuadrupedMPCROSInterface(object):
    """Name something to make it clear this is for extracting the full robot state 
    """
    def __init__(self, env_config):
        """Setup any relevant data 
        set up rospy publishers/subscribers"""
        # create robot first
        self.env = QuadrupedGymEnv(**env_config)

        self.hasStateData = False
        self.hasLegCmdData = False
        self._state_sub = rospy.Subscriber("/laikago_gazebo/lowState/state",LowState, self.lowStateCallback )
        self._legCmd_sub = rospy.Subscriber("/laikago_gazebo/legCmds",LegCmdArray, self.legCmdsCallback )

        # subscribe to get ros state
        #self._rosState_sub = rospy.Subscriber("/laikago_gazebo/ros2pybstate", PybulletState, self.getROSState)
        # get model/link states from gazebo
        #self._gazeboModelStates_sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.gazeboModelStates )
        #self._gazeboLinkStates_sub = rospy.Subscriber("/gazebo/link_states",LinkStates, self.gazeboLinkStates )
        # set model state
        #self.pub_pyb_base = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1) 

        # Service, for setting joint states (not currently working)
        #self.joint_srv = rospy.ServiceProxy("/gazebo/set_model_configuration",SetModelConfiguration)
        # Service for (un)pausing physics
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # publisher of pybullet state
        self.pub = rospy.Publisher("/laikago_gazebo/pybulletState", PybulletState, queue_size=1)
        self._rate = rospy.Rate(1000) # careful

        # publish motor states
        self.pub_motorStates = []
        for i in range(12):
            print(motor_state_strings[i])
            self.pub_motorStates.append(rospy.Publisher(motor_state_strings[i], MotorState, queue_size=1))
        

    def setGazeboState(self):
        """Just set model base pos/orn """
        base_pos = self.env._robot.GetBasePosition()
        base_orn = self.env._robot.GetBaseOrientation()

        state_msg = ModelState()
        state_msg.model_name = 'laikago_gazebo'
        state_msg.pose.position.x = base_pos[0]
        state_msg.pose.position.y = base_pos[1]
        state_msg.pose.position.z = base_pos[2]
        state_msg.pose.orientation.x = base_orn[0]
        state_msg.pose.orientation.y = base_orn[1]
        state_msg.pose.orientation.z = base_orn[2]
        state_msg.pose.orientation.w = base_orn[3]

        self.pub_pyb_base.publish(state_msg)
        self._rate.sleep()

    def setGazeboJoints(self):
        """Set model joints  """
        joint_states = list(self.env._robot.GetMotorAngles())
        print('joint states', joint_states)

        state_msg = SetModelConfigurationRequest()
        state_msg.model_name = 'laikago_gazebo' 
        state_msg.urdf_param_name = 'robot_ub18' # otherwise complains
        state_msg.joint_names = [
            "laikago_gazebo::FL_hip",
            "laikago_gazebo::FL_thigh",
            "laikago_gazebo::FL_calf",
            "laikago_gazebo::FR_hip",
            "laikago_gazebo::FR_thigh",
            "laikago_gazebo::FR_calf",
            "laikago_gazebo::RL_hip",
            "laikago_gazebo::RL_thigh",
            "laikago_gazebo::RL_calf",
            "laikago_gazebo::RR_hip",
            "laikago_gazebo::RR_thigh",
            "laikago_gazebo::RR_calf"] #gazebo_joint_names
        state_msg.joint_positions = joint_states
        res = self.joint_srv(state_msg)
        print(res)


    def pauseGazebo(self):
        res = self.pause_srv(EmptyRequest())

    def unpauseGazebo(self):
        self.unpause_srv(EmptyRequest())

    def gazeboModelStates(self,data):
        """Get gazebo model states """
        print('*'*80)
        print('gazebo model state')
        print(data)

    def gazeboLinkStates(self,data):
        """Get gazebo model states """
        print('='*80)
        print('gazebo link state')
        print(data)

    def lowStateCallback(self, data):
        """See LowState.msg for format. 
        Can also print(data)

        TODO: check body vels
        """
        self.base_pos = data.cheat.position
        self.base_orn = np.array(data.cheat.orientation)
        # quaternion is in wrong order
        #self.base_orn = np.array([self.base_orn[1],self.base_orn[2],self.base_orn[3],self.base_orn[0]])
        self.base_orn = np.roll( self.base_orn,-1)
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
        self.hasStateData = True
        #print(self.base_pos)

    def legCmdsCallback(self,data):
        """See LegCmd.msg and LegCmdArray.msg 
        Can access the data at run time when needed
        """
        print('leg cmd data',data)
        self.legCmds = data
        self.hasLegCmdData = True
        self.stepFromMPC(data)

    def stepFromMPC(self, data):
        """Take data from legCmdsCallback and step simulation.
        This is in torque control mode.
        """
        actions = np.zeros(12)

        # 
        if np.all(data.legCmds[0].pDes == 0.):
            return

        for i in range(4):
            kpCartesian = np.diag(data.legCmds[i].kpCartesianDiag)
            kdCartesian = np.diag(data.legCmds[i].kdCartesianDiag)
            u_force = data.legCmds[i].feedforwardForce
            pDes = data.legCmds[i].pDes
            vDes = data.legCmds[i].vDes
            actions[i*3:i*3+3] = self.env._robot._ComputeLegImpedanceControlActions(pDes,
                                                                                    vDes,
                                                                                    kpCartesian,
                                                                                    kdCartesian,
                                                                                    u_force,
                                                                                    i #legID
                                                                                    )
        self.env.step(actions)


    def sendMotorStates(self):
        """Send motor states """
        joint_pos = self.env._robot.GetMotorAngles()
        joint_vel = self.env._robot.GetMotorVelocities()
        joint_torques = self.env._robot.GetMotorTorques()
        for i in range(12):
            data = MotorState()
            data.position = joint_pos[i]
            data.velocity = joint_vel[i]
            data.torque = joint_torques[i]
            self.pub_motorStates[i].publish(data)



    def sendPybData(self,v_des=[1,0,0],yaw_rate_des=0.):
        """Format data into ROS format. 
        TODO: Get this info from info {} in QuadrupedGymEnv.

        Here is what the MPC needs:

        struct StateEstimate {
            Vec3<double> position;
            Quat<double> orientation;
            Vec3<double> rpy;
            RotMat<double> rBody;
            Vec3<double> vBody;
            Vec3<double> omegaBody;
            Vec3<double> vWorld;
            Vec3<double> omegaWorld;
            
            Vec4<double> contactEstimate;
            Vec3<double> aBody, aWorld;
        }
        // and also desired commands:
           Vec3<double> v_des;
            double yaw_rate_des;
        """
        data = PybulletState()
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

        data.position = pos
        data.orientation = orn
        data.rpy = rpy
        data.rBody = R.T.ravel() # numpy row indexed, eigen column indexed
        data.vBody = vBody
        data.omegaBody = omegaBody
        data.vWorld = vWorld
        data.omegaWorld = omegaWorld
        data.v_des = v_des
        data.yaw_rate_des = yaw_rate_des
        # TODO? add contactEstimate, aBody, aWorld
        #print('*'*30, 'publishing PybulletState:')
        #print(data)
        self.pub.publish(data)
        self.sendMotorStates()
        #print('*'*30, 'Done')
        self._rate.sleep()

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

        # print('*'*80, '\n COMPARING ROS WITH CURRENT PYB, should be close-ish')
        # print('ros pos',data.position)
        # print('pyb pos',pos)
        # print('ros orn', data.orientation)
        # print('pyb or', orn)
        # print('ros rpy',data.rpy)
        # print('pyb rpy', rpy)
        # print('ros rBody',data.rBody)
        # print('pyb rBody', R.ravel())
        # print('ros vBody',data.vBody)
        # print('pyb vBody', vBody)
        # print('ros omegaBody',data.omegaBody)
        # print('pyb omegaBody',omegaBody)
        # print('ros vWorld',data.vWorld)
        # print('pyb vWorld', vWorld)
        # print('ros omegaWorld',data.omegaWorld)
        # print('pyb omegaWorld',omegaWorld)




####################################################
# Setup ros node, listener
####################################################
env_config = {
    "isRLGymInterface": False, 
    "robot_config": robot_config, 
    "time_step": 0.001, # make sure this is small enough
    "render": True, 
    "motor_control_mode": "TORQUE"
}

_rosBridge = QuadrupedMPCROSInterface(env_config)

time.sleep(1)

counter = 0

while True:
    # send robot state to ROS
    _rosBridge.sendPybData(v_des=[0,0,0])
    # make sure we get a result back from ROS
    time.sleep(0.1)

    # should really iterate through legCmds
    if _rosBridge.env.is_fallen():
        _rosBridge.env.reset()

    counter += 1
