"""Example file, self-contained.

Send data to ROS, run MPC, returns leg commands to step pybullet environment.

Need 3 terminals, with in order:

Terminal 1 (after source ROS path):
$ roslaunch laikago_gazebo aliengo.launch
Terminal 2 (after source ROS path):
$ rosrun laikago_gazebo pyb_mpc
Terminal 3 [this one] (source ROS path, activate virtualenv):
$ python run_mpc.py

"""
import time
import numpy as np
# ros imports
import rospy
from rospy.numpy_msg import numpy_msg
# get relevant msg formats
from laikago_msgs.msg import PybulletState, PybulletFullState
from laikago_msgs.srv import MPCService, MPCServiceRequest, MPCServiceResponse
from std_msgs.msg import Empty, Int32
# pybullet env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# get appropriate config file
import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
import usc_learning.envs.quadruped_master.configs_a1 as robot_config
# utils
from usc_learning.utils.utils import nicePrint

""" 
Performance Notes:
+ Normal force up to -242N for trotting
+ Gaits - success depends a lot on the initial condition, where ROS initial condition is almost falling over in pybullet
    -starting from standing at ~0.4m height
    -trotting good
    -pronking good
    -walking ok
    -galloping doesn't fall but does not look optimal
    -bounding/pacing not working, 
    
"""
rospy.init_node('pyb_interface', anonymous=True)


####################################################
# ROS MPC Interface
####################################################
class QuadrupedMPCROSInterface(object):
    """ Quadruped ROS MPC Interface """
    def __init__(self, env_config):
        """Setup env, ros communication """
        # quadruped env
        self.env = QuadrupedGymEnv(**env_config)

        # service client to call MPC with pyb state, and return leg commands
        self._rate = rospy.Rate(1000) # careful
        rospy.wait_for_service('/laikago_gazebo/mpc_comm')
        try:
            self.mpc_comm = rospy.ServiceProxy('/laikago_gazebo/mpc_comm', MPCService,persistent=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        # rospy.wait_for_service('/laikago_gazebo/balance_comm')
        # try:
        #     self.balance_comm = rospy.ServiceProxy('/laikago_gazebo/balance_comm', MPCService,persistent=True)
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)


        # publishers - reset MPC, and set gait
        self._resetMPC_pub = rospy.Publisher("/laikago_gazebo/reset_mpc", Empty, queue_size=1)
        self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)

    def stepFromMPC(self, data, calcTorque=True):
        """Take data from LegCmdArray and step simulation (torque control mode). """
        # torques calculated from ROS
        tau_actions = np.array(data.motorCmd)

        all_Fns = np.zeros(12)
        all_pDes = np.zeros(12)
        all_vDes = np.zeros(12)
        all_kp = np.zeros(12)

        R = self.env._robot.GetBaseOrientationMatrix()
        
        # test if we can use the commands for more than 1 timestep
        for _ in range(1):
            if calcTorque:
                actions = np.zeros(12)
                for i in range(4):
                    kpCartesian = np.diag(data.legCmds[i].kpCartesianDiag)
                    kdCartesian = np.diag(data.legCmds[i].kdCartesianDiag)
                    u_force = data.legCmds[i].feedforwardForce
                    pDes = data.legCmds[i].pDes
                    vDes = data.legCmds[i].vDes
                    # test using different constant kp or Fn only 
                    #kpCartesian = np.diag([500,500,100])
                    #kdCartesian = np.diag([30,30,30])
                    #u_force = np.array([0,0,u_force[2]]) # just z
                    #vDes = np.zeros(3)
                    all_Fns[i*3:i*3+3] = R.T @ u_force
                    all_pDes[i*3:i*3+3] = pDes
                    all_vDes[i*3:i*3+3] = vDes
                    all_kp[i*3:i*3+3] = kpCartesian.diagonal()
                    actions[i*3:i*3+3] = self.env._robot._ComputeLegImpedanceControlActions(pDes,
                                                                                            vDes,
                                                                                            kpCartesian,
                                                                                            kdCartesian,
                                                                                            u_force,
                                                                                            i #legID
                                                                                            )
            #     self.env.step(actions)
            # else:
            #     self.env.step(tau_actions)  
            #     break

        if self.env._motor_control_mode == "TORQUE":
            actions = self._unscale_helper(actions,
                                            -self.env._robot_config.TORQUE_LIMITS,
                                            self.env._robot_config.TORQUE_LIMITS)
            self.env.step(actions)
        elif self.env._motor_control_mode == "IMPEDANCE_FULL":
            print('---------------------------------------- TORQUE')
            print('pos', all_pDes)
            print('vel', all_vDes)
            print('ffwd', all_Fns)
            print('kpCa', all_kp)
            print('-----------torque', actions)
            MPC_HIGH = np.array([ 1, 1, 0]*4 + [ 3]*12 + [ 300]*12)
            MPC_LOW =  np.array([-1,-1,-1]*4 + [-3]*12 + [-300]*12)
            imp_full_act = np.concatenate((all_pDes,all_vDes, all_Fns))
            imp_full_act = self._unscale_helper(imp_full_act,MPC_LOW,MPC_HIGH)
            self.env.step(imp_full_act)
        #print(tau_actions)
        # compare calculated torques and those from ROS
        #print('*'*50)
        #nicePrint(all_Fns)
        #print(self.env._robot.GetBaseLinearVelocity())
        #self.env.step(tau_actions)  
        #self.env.step(actions) 
    def _unscale_helper(self, u, lower_lim, upper_lim):
        """Scale down to [-1,1] from lower/upper limits. 
        This needs to be made general in case action range is not [-1,1]
        """
        action = (2*u - (upper_lim + lower_lim)) / (upper_lim - lower_lim)
        #new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        # verify clip
        return np.clip(action, -1, 1)



    def getAndSetActionFromMPC(self):
        """Get action, v_des should be based on robot current velocity. 
        gradually increase robot speed, MPC assumes max vel of 2 m/s for trotting
        """
        curr_vel_x = self.env._robot.GetBaseLinearVelocity()[0]
        # may want some more logic here
        v_des = [ min(curr_vel_x+.1, 5), 0,0]
        yaw_rate_des = -1* self.env._robot.GetBaseOrientationRollPitchYaw()[2]
        pyb_state = self.getPybData(v_des = v_des, yaw_rate_des=yaw_rate_des)
        # send pybullet states to ROS and return leg commands
        leg_cmds = self.mpc_comm(pyb_state)
        print(leg_cmds)
        # step sim
        self.stepFromMPC(leg_cmds.legCmdArray)

    def getAndSetActionFromBalanceController(self):
        """Get action, v_des should be based on robot current velocity. 
        gradually increase robot speed, MPC assumes max vel of 2 m/s for trotting
        """
        #curr_vel_x = self.env._robot.GetBaseLinearVelocity()[0]
        # may want some more logic here
        #v_des = [ min(curr_vel_x+.1, 5), 0,0]
        #yaw_rate_des = -1* self.env._robot.GetBaseOrientationRollPitchYaw()[2]
        pyb_state = self.getPybData()
        # send pybullet states to ROS and return leg commands
        leg_cmds = self.balance_comm(pyb_state)
        #print(leg_cmds)
        print(pyb_state)
        # step sim
        self.stepFromMPC(leg_cmds.legCmdArray, calcTorque=True)

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

    def getPybData(self,v_des=[1,0,0],yaw_rate_des=0.):
        """Format data into ROS format. 
        TODO: get this from step return info {} in QuadrupedGymEnv.

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
            Vec3<double> v_des;
            double yaw_rate_des;
        """
        #data = PybulletState()
        data = PybulletFullState()
        # position
        pos = self.env._robot.GetBasePosition()
        # orientation
        orn = self.env._robot.GetBaseOrientation()
        orn =  np.roll(orn,1) #[orn[3], orn[0], orn[1], orn[2]]
        # rpy, rBody
        rpy = self.env._robot.GetBaseOrientationRollPitchYaw()
        # Note this is row major, unlike Eigen
        R = self.env._robot.GetBaseOrientationMatrix()
        # world lin/ang vel
        vWorld = self.env._robot.GetBaseLinearVelocity()
        omegaWorld = self.env._robot.GetBaseAngularVelocity()
        # body lin/ang vel
        vBody = R @ vWorld
        omegaBody = self.env._robot.GetTrueBaseRollPitchYawRate()
        # joint states
        joint_pos = self.env._robot.GetMotorAngles()
        joint_vel = self.env._robot.GetMotorVelocities()
        joint_torques = self.env._robot.GetMotorTorques()

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
        # set joint states
        for i in range(12):
            data.motorState[i].position = joint_pos[i]
            data.motorState[i].velocity = joint_vel[i]
            data.motorState[i].torque = joint_torques[i]

        return data


####################################################
# Create env and run
####################################################
env_config = {
    "isRLGymInterface": True, 
    "robot_config": robot_config, 
    "time_step": 0.001, # make sure this is small enough
    "action_repeat": 1,
    "render": True, 
    "motor_control_mode": "IMPEDANCE_FULL"
}

_rosBridge = QuadrupedMPCROSInterface(env_config)
time.sleep(1)
counter = 0
while True:
    # send robot state to ROS
    _rosBridge.getAndSetActionFromMPC()
    #_rosBridge.getAndSetActionFromBalanceController()
    # if fallen, reset and choose random new gait
    if counter > 10000 or abs(_rosBridge.env._robot.GetBaseOrientationRollPitchYaw()[0]) > 1: 
        _rosBridge.resetMPC()
        _rosBridge.setGait(2)#int(np.random.randint(6)+1))
        _rosBridge.env.reset()
        counter = 0

    #_rosBridge._rate.sleep()
    counter += 1