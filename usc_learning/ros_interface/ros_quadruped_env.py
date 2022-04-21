"""
Use MPC or Balance Controller as part of RL. 

Must be used as one process for now (TODO: set up additional communication for MPI)


To run, need 3 terminals, with in order:

Terminal 1 (after source ROS path):
$ roslaunch laikago_gazebo aliengo.launch
Terminal 2 (after source ROS path):
$ rosrun laikago_gazebo pyb_mpc
Terminal 3 [this one] (source ROS path, activate virtualenv):
[run either rllib or stable baselines]


"""
import os, sys
import time
import numpy as np
# ros imports
import rospy
from rospy.numpy_msg import numpy_msg
# get relevant msg formats
from laikago_msgs.msg import PybulletState, PybulletFullState, LowState, LegCmdArray
from laikago_msgs.srv import MPCService, MPCServiceRequest, MPCServiceResponse
#from laikago_msgs.srv import RLService, RLServiceRequest, RLServiceResponse
from std_msgs.msg import Empty, Int32
#from std_srvs.srv import Empty, EmptyRequest
#from std_msgs.msg import Empty as EmptyMsg
#import std_srvs.srv.Empty

# pybullet env
# from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
import usc_learning.envs.quadruped_master.quadruped_gym_env as quadruped_gym_env
# get appropriate config file
#import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
import usc_learning.envs.quadruped_master.configs_a1 as robot_config
# utils
from usc_learning.utils.utils import nicePrint

import gym
from gym import spaces


ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
""" 
Notes:
    
"""
rospy.init_node('pyb_interface', anonymous=True)

# task modes
"""
either 
1) MPC_FOOT_POS: choose foot positions for MPC


2) BALANCE_NO_FALL: use Balance Controller to not fall down during training
    -i.e. takes over when detecting fall

"""

class ROSQuadrupedGymEnv(quadruped_gym_env.QuadrupedGymEnv):
    """ Quadruped ROS MPC Interface 

    Use this to get MPC or balance controller actions
    """
    def __init__(self,
            robot_config=robot_config,
            isRLGymInterface=True,
            energy_weight=0.0,
            task_mode="BALANCE_NO_FALL",
            accurate_motor_model_enabled=True,
            motor_control_mode="IMPEDANCE_XZ_ONLY",
            task_env="FWD_LOCOMOTION",
            observation_space_mode="DEFAULT_CONTACTS",
            rl_action_high=1.0,       # action high for RL
            grow_action_space=False,  # use growing action spaces
            hard_reset=True,
            render=False,
            record_video=False,
            env_randomizer=None,
            randomize_dynamics=False,
            add_terrain_noise=False,
            **kwargs
        ):#self, env_opt, model=None):
        """Setup env, ros communication 
        Argument is either:
            -a dictionary, for env configuration,
            -created env from RL
        """
        self._task_mode = task_mode
        self._setup_ros_comm()
        if task_mode == "MPC_FOOT_POS":
            motor_control_mode = "TORQUE" # b/c no additional processing later
            time_step = 0.001
            action_repeat = 30
        elif task_mode == "BALANCE_NO_FALL":
            time_step = 0.001
            action_repeat = 10
        else:
            raise ValueError('task mode not implemented')

        super(ROSQuadrupedGymEnv, self).__init__(
            #urdf_root=pybullet_data.getDataPath(), #pyb_data.getDataPath() # careful_here 
            robot_config=robot_config,
            isRLGymInterface=isRLGymInterface, 
            energy_weight=energy_weight,
            time_step=time_step,
            action_repeat=action_repeat,
            accurate_motor_model_enabled=accurate_motor_model_enabled,
            motor_control_mode=motor_control_mode,
            task_env=task_env,
            observation_space_mode=observation_space_mode,
            rl_action_high=rl_action_high,  
            grow_action_space=grow_action_space,
            hard_reset=hard_reset,
            render=render,
            record_video=record_video,
            env_randomizer=env_randomizer,
            randomize_dynamics=randomize_dynamics,
            add_terrain_noise=add_terrain_noise) 


    ######################################################################################
    # obs/action spaces, reset
    ######################################################################################
    def setupObsAndActionSpaces(self):
        """Set up observation and action spaces for RL. """
        # Observation space
        if self._observation_space_mode == "DEFAULT":
          observation_high = (self._robot.GetObservationUpperBound() + OBSERVATION_EPS)
          observation_low =  (self._robot.GetObservationLowerBound() - OBSERVATION_EPS)
        elif self._observation_space_mode == "DEFAULT_CONTACTS":
          observation_high = (self._robot.GetObservationContactsUpperBound() + OBSERVATION_EPS)
          observation_low =  (self._robot.GetObservationContactsLowerBound() - OBSERVATION_EPS)
        elif self._observation_space_mode == "EXTENDED":
          observation_high = (self._robot.GetExtendedObservationUpperBound() + OBSERVATION_EPS)
          observation_low =  (self._robot.GetExtendedObservationLowerBound() - OBSERVATION_EPS)
        else:
          raise ValueError("observation space not defined or not intended")
        # Action space
        if self._motor_control_mode in ["IMPEDANCE_XZ_ONLY"]:
            action_dim = 8
        elif self._task_mode in ['MPC_FOOT_POS']:
            action_dim = 8 # 4 xy positions
        elif self._motor_control_mode in ["PD","TORQUE", "IK", "IMPEDANCE_NO_Y"]:
          action_dim = 12
        elif self._motor_control_mode in ["IMPEDANCE"]:
          action_dim = 16 # (xyz pos + Fn)*4
        # elif self._motor_control_mode in ["IMPEDANCE_FORCE"]:
        #   action_dim = 4 # Fn*4
        elif self._motor_control_mode in ["IMPEDANCE_GAINS"]:
          action_dim = 22 # (xyz pos + Fn)*4 + 3 kp_gains + 3 kd_gains
        elif self._motor_control_mode in ["IMPEDANCE_3D_FORCE"]:
          action_dim = 24 # (xyz pos + 3D contact force)*4  
        else:
          raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")

        if self._GAS: # likely in lower range than [-1,1]
            action_high = np.array([self._rl_action_bound] * action_dim)
        else:
            action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

    def reset(self):
        """reset, call super and reset any necessary MPC/balance controller """

        # reset any MPC or balance controller needed things
        print('Reset after ',self.get_sim_time(), '(s)')
        try:
            print( 'pos:', self._robot.GetBasePosition())
        except:
            pass
        self.resetMPC()
        self.resetBalanceController()

        noisy_obs = super(ROSQuadrupedGymEnv, self).reset()
        self._starting_ee_pos = np.concatenate(([self._robot.GetLocalEEPos(i) for i in range(4)]))
        return noisy_obs #super(ROSQuadrupedGymEnv, self).reset()

    ######################################################################################
    # step
    ######################################################################################
    def _transform_action_to_motor_command(self, action):
        """Transform action, depending on mode. """
        if self._task_mode == "MPC_FOOT_POS":
            #print('TODO: scale action to necessary position limits')
            action = self._ScaleActionToFootPosRange(action)
        elif self._motor_control_mode == "PD":
            action = self._robot.ScaleActionToPosRange(action)
        elif self._motor_control_mode == "IK":
            action = self._robot.ScaleActionToIKRange(action)
        elif self._motor_control_mode in ["IMPEDANCE", "IMPEDANCE_FORCE","IMPEDANCE_GAINS","IMPEDANCE_3D_FORCE"]:
            action, _ = self._robot.ScaleActionToImpedanceControl(action)
        elif self._motor_control_mode == "TORQUE":
            action = self._robot.ScaleActionToTorqueRange(action)
        else:
            raise ValueError("motor control mode" + self._motor_control_mode + "not implemented yet.")
        return action

    def step(self, action):
        """ Step forward the simulation, given the action. 

        MPC mode: continuously get actions for duration of one gait cycle
        Balance: get balance contorller action if detect falling

        """
        if self._is_render:
            self._render_step_helper()

        self.num_non_rl_actions = 0

        # for MPC, want to iterate through a full cycle, but terminate if bad
        # for balance controller, act normal, just dectect if falling, and step in
        for _ in range(self._action_repeat):
            # only scale action if it's a gym interface
            #if self._isRLGymInterface: # this is doing the same as _leg_model_enabled
            if self._GAS:
                proc_action = self._transform_action_GAS(action.copy())
            else:
                proc_action = self._transform_action_to_motor_command(action.copy())
            proc_action = self._get_action_ros(proc_action)

            self._robot.ApplyAction(proc_action)
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            if self._is_render:
                self._render_step_helper()

        self._env_step_counter += 1
        reward = self._reward()


        # for balance controller, have two terminates
        # 1 - when to activate balance (i.e. while angular velocity is > threshold, or base COM not in support polygon)
        # 2 - actually fell over (hopefully never happens)

        # IDEA: use the action from the balance controller to update the NNs (impedance? torque?)

        done = self._termination() or self.get_sim_time() > self._MAX_EP_LEN

        # just assume balance controller would work,  otherwise this messes up advantage computation
        if self.num_non_rl_actions > 0:
            done = True
        return np.array(self._noisy_observation()), reward, done, {'num_non_rl_actions': self.num_non_rl_actions} 

    def _get_action_ros(self,action):
        """Helper to get action from ROS (if needed) """
        if self._task_mode == "MPC_FOOT_POS":
            return self.getActionFromMPC(action)
        elif self._task_mode == "BALANCE_NO_FALL":
            if True: # use balance controller as intended
                #return self.getActionFromBalanceController()
                #return self.getActionFromMPC(None)
                if self._need_balance_controller(angVelThreshold=1.5):
                    self.num_non_rl_actions += 1
                    #print('ee  now', np.concatenate(([self._robot.GetLocalEEPos(i) for i in range(4)])))
                    #print('ee orig', self._starting_ee_pos)
                    #print(np.linalg.norm(np.concatenate(([self._robot.GetLocalEEPos(i) for i in range(4)])) - self._starting_ee_pos))
                    if np.linalg.norm(np.concatenate(([self._robot.GetLocalEEPos(i) for i in range(4)])) - self._starting_ee_pos) > 0.5:# was 0.1
                        # check if feet are too close to body COM, then use PD controller on init motor angles otherwise
                        #print('init torques')

                        motor_torques = -1 * (self._robot_config.MOTOR_KP * (self._robot.GetMotorAngles() - self._robot_config.INIT_MOTOR_ANGLES)) \
                                            - self._robot_config.MOTOR_KD * (self._robot.GetMotorVelocities())
                        return motor_torques
                    else:
                        #print('balance')
                        return self.getActionFromBalanceController()
                else:
                    self.resetBalanceController()
                    return action

            #return self.getActionFromBalanceController()

            if self._need_balance_controller(angVelThreshold=0.5): # while this is true, we should continue to get actions from balance controller
                if self._need_balance_controller(angVelThreshold=1):
                    return self.getBalanceActionFromMPC()
                else:
                    return self.getActionFromBalanceController()
                    
            else:
                #return self.getActionFromBalanceController()
                # not falling, just use normal action, and reset balance controller in case need again
                #print('use rl')
                self.resetBalanceController()
                self.resetMPC()
                return action
        else:
            raise ValueError('invalid task mode')

    def _ScaleActionToFootPosRange(self,actions):
        """Scale actions to foot position ranges """
        #print('Check what relative positions MPC is using')
        low_lim =  self._robot_config.IK_POS_LOW_MPC
        upp_lim =  self._robot_config.IK_POS_UPP_MPC
        u = np.clip(actions,-1,1)
        #print('clipped u', u)
        u = self._robot._scale_helper(u, low_lim, upp_lim) 
        #print('low leg', self._robot_config.IK_PO)
        # print('low_lim',low_lim)
        # print('upp lim', upp_lim)
        # print(u)
        return u

    def _need_balance_controller(self, upDiff=0.99, angVelThreshold=1): # 0.9, 0.3
        """ See if we're falling over, i.e. need balance controller. """

        """Decide whether the quadruped has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the quadruped is considered fallen.

        Returns:
          Boolean value that indicates whether the quadruped has fallen.
        """
        orientation = self._robot.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self._robot.GetBasePosition()
        ang_vel = self._robot.GetTrueBaseRollPitchYawRate()#[:2]# self._robot.GetBaseAngularVelocity()

        # just check height and orientation
        orn = self._robot.GetBaseOrientationRollPitchYaw()
        return abs(orn[0]) > 0.08 or abs(orn[1]) > 0.08 or pos[2] < self._robot_config.IS_FALLEN_HEIGHT

        #print(ang_vel)
        # return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < 0.13) # from minitaur
        # return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < upDiff \
        #         or pos[2] < self._robot_config.IS_FALLEN_HEIGHT) \
        #         or np.linalg.norm(ang_vel) > angVelThreshold

    def _termination(self):
        """Should terminate episode """
        return self.is_fallen() #self._need_balance_controller(upDiff=0.85,angVelThreshold=10) 

    ######################################################################################
    #
    # Communication, callbacks, etc.
    #
    ######################################################################################

    def _setup_ros_comm(self):
        """Setup and necessary ROS comm. """

        ###################################### MPC ######################################
        rospy.wait_for_service('/laikago_gazebo/mpc_comm')
        try:
            self.mpc_comm = rospy.ServiceProxy('/laikago_gazebo/mpc_comm', MPCService,persistent=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        # publishers - reset MPC, and set gait
        self._resetMPC_pub = rospy.Publisher("/laikago_gazebo/reset_mpc", Empty, queue_size=1)
        self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)
        #self._setRLActions_pub = rospy.Publisher("/laikago_gazebo/mpc_set_rl_foot_pos",)
        # add desired to MPCService aka PybulletFullState as part of MPC sercice

        ############################### Balance Controller ###############################
        rospy.wait_for_service('/laikago_gazebo/balance_comm')
        try:
            self.balance_comm = rospy.ServiceProxy('/laikago_gazebo/balance_comm', MPCService,persistent=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        # publishers
        self._resetBalance_pub = rospy.Publisher("/laikago_gazebo/reset_balance", Empty, queue_size=1)

        ############################### gazebo #(needed?) ###############################

        # rate (not used)
        self._rate = rospy.Rate(1000) # careful


    ######################################################################################
    # ROS state utils
    ######################################################################################
    def convertToTorquesFromROS(self, data, calcTorque=True):
        """Take data from LegCmdArray and step simulation (torque control mode). """
        # torques calculated from ROS
        if not calcTorque:
            tau_actions = np.array(data.motorCmd)
            return tau_actions

        #all_Fns = np.zeros(12)
        #all_pDes = np.zeros(12)

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
                    #all_Fns[i*3:i*3+3] = u_force
                    #all_pDes[i*3:i*3+3] = pDes
                    actions[i*3:i*3+3] = self._robot._ComputeLegImpedanceControlActions(pDes,
                                                                                        vDes,
                                                                                        kpCartesian,
                                                                                        kdCartesian,
                                                                                        u_force,
                                                                                        i #legID
                                                                                        )
                #self.env.step(actions)
            else:
                #self.env.step(tau_actions)  
                break
        
        #print('returned actions', actions)
        return actions

    def getPybData(self,v_des=[1,0,0],yaw_rate_des=0.):
        """Format data into ROS format. 

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
        pos = self._robot.GetBasePosition()
        # orientation
        orn = self._robot.GetBaseOrientation()
        orn =  np.roll(orn,1) #[orn[3], orn[0], orn[1], orn[2]]
        # rpy, rBody
        rpy = self._robot.GetBaseOrientationRollPitchYaw()
        # Note this is row major, unlike Eigen
        R = self._robot.GetBaseOrientationMatrix()
        # world lin/ang vel
        vWorld = self._robot.GetBaseLinearVelocity()
        omegaWorld = self._robot.GetBaseAngularVelocity()
        # body lin/ang vel
        vBody = R @ vWorld
        omegaBody = self._robot.GetTrueBaseRollPitchYawRate()
        # joint states
        joint_pos = self._robot.GetMotorAngles()
        joint_vel = self._robot.GetMotorVelocities()
        joint_torques = self._robot.GetMotorTorques()

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

    ######################################################################################
    # MPC and Balance Controller
    ######################################################################################
    def getActionFromMPC(self,feetPos):
        """Get action, v_des should be based on robot current velocity. 
        gradually increase robot speed, MPC assumes max vel of 2 m/s for trotting

        TODO: allow RL to choose velocity (and yaw rate?)
        """
        curr_vel_x = self._robot.GetBaseLinearVelocity()[0]
        # may want some more logic here
        v_des = [ min(curr_vel_x+.1, 5), 0,0]
        yaw_rate_des = -1* self._robot.GetBaseOrientationRollPitchYaw()[2]
        pyb_state = self.getPybData(v_des = v_des, yaw_rate_des=yaw_rate_des)
        # if self._task_mode == "MPC_FOOT_POS":
        # add desired foot positions from RL
        pyb_state.rlFootPos = feetPos
        pyb_state.rlFlag = 1
        # send pybullet states to ROS and return leg commands
        leg_cmds = self.mpc_comm(pyb_state)
        # convert to torques
        return self.convertToTorquesFromROS(leg_cmds.legCmdArray, calcTorque=True)

    def getBalanceActionFromMPC(self,v_des=[0,0,0],yaw_rate_des=0):
        """For balancing, just don't fall over 
        v_des should be based on robot current velocity, but probably want to be 0 

        """
        v_des = [max(self._robot.GetBaseLinearVelocity()[0],0),0,0]
        pyb_state = self.getPybData(v_des=v_des, yaw_rate_des=yaw_rate_des)
        # send pybullet states to ROS and return leg commands
        leg_cmds = self.mpc_comm(pyb_state)
        # convert to torques
        return self.convertToTorquesFromROS(leg_cmds.legCmdArray, calcTorque=True)

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

    def getActionFromBalanceController(self):
        """Get balance controller actions from ROS
        TODO: check balance controller firstRun reset flag (when to activate?)
            -should change anything about desired position(s)?
        """
        # get pybullet full state
        pyb_state = self.getPybData()
        # send pybullet states to ROS and return leg commands
        leg_cmds = self.balance_comm(pyb_state)
        #print(leg_cmds)
        # convert to torques
        return self.convertToTorquesFromROS(leg_cmds.legCmdArray, calcTorque=True)

    def resetBalanceController(self):
        """Reset Balance Controller """
        self._resetBalance_pub.publish(Empty())
