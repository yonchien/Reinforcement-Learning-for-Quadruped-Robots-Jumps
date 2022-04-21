"""
Get robot state from ROS, get action from policy network, send back leg commands to ROS.

Check: is this 
    -a latency issue (cannot update ROS sim fast enough) 
    -a transferability issue (policy does not generalize to different environment)

Need 3 terminals, with in order:

Terminal 1 (after source ROS path):
$ roslaunch laikago_gazebo aliengo.launch
Terminal 2 (after source ROS path):
$ rosrun laikago_gazebo pyb_mpc
Terminal 3 [this one] (source ROS path, activate virtualenv):
$ python rl_ros_interface.py


This is going to get into a mess, how best to merge this with rllib interface?

This file: checking stable baselines
also: have interface to interact between both

"""
import os, sys
import time
from datetime import datetime
import numpy as np
# ros imports
import rospy
from rospy.numpy_msg import numpy_msg
# get relevant msg formats
from laikago_msgs.msg import PybulletState, PybulletFullState, LowState, LegCmdArray
from laikago_msgs.msg import RLObs, RLAct
from laikago_msgs.srv import MPCService, MPCServiceRequest, MPCServiceResponse
from laikago_msgs.srv import RLService, RLServiceRequest, RLServiceResponse
from std_msgs.msg import Empty, Int32
from std_srvs.srv import Empty, EmptyRequest
from std_msgs.msg import Empty as EmptyMsg
#import std_srvs.srv.Empty
# pybullet env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# get appropriate config file
import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
#import usc_learning.envs.quadruped_master.configs_a1 as robot_config
# utils
from usc_learning.utils.utils import nicePrint
from usc_learning.utils.robot_utils import QuaternionToOrientationMatrix
from usc_learning.utils.utils import plot_results, load_rllib
from usc_learning.utils.file_utils import get_latest_model_rllib, get_latest_directory, read_env_config

# RL stuff
import pickle
import gym
from collections.abc import Mapping
import ray
from ray.rllib.agents import ppo 
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
""" 
Notes:
    
"""
rospy.init_node('pyb_interface', anonymous=True)


USE_RLLIB = True
########################################################################################################
# Load RL stuff... this needs to be in its own file/class
########################################################################################################
USING_VEC_ENV = True
alg = "PPO"

# ray_path = "/home/guillaume/Documents/Research/DRL/Reinforcement-Learning-for-Quadruped-Robots/usc_learning/learning/rllib"
# log_dir = os.path.join(ray_path, 'ray_results','071220024610')

######################################################################

from usc_learning.utils.learning_utils import get_stable_baselines_model_env, get_rllib_model_env

if not USE_RLLIB:
    # stable baselines
    interm_dir = "/home/guillaume/Documents/Research/DRL/Reinforcement-Learning-for-Quadruped-Robots/usc_learning/learning/logs/intermediate_models/"
    log_dir = interm_dir + '071020014919/' # A1
    log_dir = interm_dir + '072120143907' # standing
    log_dir = interm_dir + '072120173035'
    model, env = get_stable_baselines_model_env(log_dir)

else:
    # rllib
    ray.init()
    interm_dir = "/home/guillaume/Documents/Research/DRL/Reinforcement-Learning-for-Quadruped-Robots/usc_learning/learning/rllib/ray_results/"
    log_dir = interm_dir + '072020220449/' # standing task
    log_dir = interm_dir + '072420193153/' # standing (new)
    #log_dir = interm_dir + '072420181443/'
    #log_dir = interm_dir + '072020232048'
    model, env = get_rllib_model_env(log_dir)

########################################

########################################################################################################
# RL Interface, this also needs to be in its own file/class
# Could merge DummyQuadrupedEnv with the below into
# ROSEnv, which contains the VecEnv/normal quadruped env, and all the communication related stuff
########################################################################################################
class QuadrupedRLROSInterface(object):
    """ Quadruped ROS MPC Interface """
    def __init__(self, env_opt, model=None):
        """Setup env, ros communication 
        Argument is either:
            -a dictionary, for env configuration,
            -created env from RL
        """
        # quadruped env
        #print( type(env_opt))
        if isinstance(env_opt, Mapping):
            self.env = QuadrupedGymEnv(**env_opt)
        else:
            if USE_RLLIB:
                # a vecnormalized quadruped gym env
                self.env = env_opt
                self.model = model
                self.venv = self.env.env.env.env
                self.quadruped_env = self.env.env.env.env.venv.envs[0].env #lol

                print(self.env)                                 # rllib Monitor
                print(self.env.env)                             # rllib NormalizeActionWrapper
                print(self.env.env.env)                         # Dummy Quadruped Instance
                print(self.env.env.env.env)                     # stable_baselines.common.vec_env.vec_normalize.VecNormalize
                print(self.env.env.env.env.venv)                # dummy vec env
                print(self.env.env.env.env.venv.envs[0].env)    # Quadruped Gym Env

            else:
                # stable baselines
                self.env = env_opt
                self.model = model
                self.quadruped_env = self.env.envs[0].env



        self.env.reset()

        side_sign = [-1, 1, -1, 1];
        legCmdArray = LegCmdArray()
        for i in range(4):
            legCmdArray.legCmds[i].pDes = (0, side_sign[i]*0.083 , -0.3)#info['pDes'][i*3:i*3+3]
            legCmdArray.legCmds[i].vDes = (0,0,0) #info['vDes'][i*3:i*3+3]
            legCmdArray.legCmds[i].kpCartesianDiag = (500,500,1000) #info['kpCartesianDiag'][i*3:i*3+3]
            legCmdArray.legCmds[i].kdCartesianDiag = (30,30,30)  #info['kdCartesianDiag'][i*3:i*3+3]
            legCmdArray.legCmds[i].feedforwardForce = (0,0,0) #info['feedforwardForce'][i*3:i*3+3]

        self._action_space_low = np.concatenate((self.quadruped_env._robot_config.IK_POS_LOW_LEG, self.quadruped_env._robot_config.MIN_NORMAL_FORCE))
        self._action_space_upp = np.concatenate((self.quadruped_env._robot_config.IK_POS_UPP_LEG, self.quadruped_env._robot_config.MAX_NORMAL_FORCE))

        self.legCmdArray = legCmdArray

        self.rl_act = RLAct()
        # sys.exit()
        # pause/unpause physics
        self.pause_srv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_srv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        #print('set up pause/unpause')

        # don't do this
        #self.pauseGazebo()
        #print('post pause')

        # subscribers
        # self._state_sub = rospy.Subscriber("/laikago_gazebo/lowState/state",LowState, self.lowStateCallback )
        # service to accept state from ROS, and return actions
        #self._state_service = rospy.Service("/laikago_gazebo/rl_comm", RLService, self.rlServiceCallback)

        # publish - legCmdArray
        self._legCmd_pub = rospy.Publisher("/laikago_gazebo/rl_leg_cmds",LegCmdArray, queue_size=1 )

        # RL
        self._rl_obs_sub = rospy.Subscriber("/laikago_gazebo/rl_obs", RLObs, self.receiveObsCallback )
        self._rl_act_pub = rospy.Publisher("/laikago_gazebo/rl_act", RLAct, queue_size=1 )

        self._rate = rospy.Rate(1000) # careful

        # # service client to call MPC with pyb state, and return leg commands
        # 
        # rospy.wait_for_service('/laikago_gazebo/mpc_comm')
        # try:
        #     self.mpc_comm = rospy.ServiceProxy('/laikago_gazebo/mpc_comm', MPCService,persistent=True)
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)

        # # publishers - reset MPC, and set gait
        # self._resetMPC_pub = rospy.Publisher("/laikago_gazebo/reset_mpc", Empty, queue_size=1)
        # self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)

        # publish - turn on
        # self._python_available_pub = rospy.Publisher("/laikago_gazebo/python_available",EmptyMsg, queue_size=20 )
        # print(EmptyMsg())
        # # ros not receiving - why is this?
        # for _ in range(10):
        #     self._python_available_pub.publish(EmptyMsg())
        #     self._rate.sleep()

    def sendAvailable(self):
        print(EmptyMsg())
        #for _ in range(10):
        self._python_available_pub.publish(EmptyMsg())
        self._rate.sleep()

    def pauseGazebo(self):
        res = self.pause_srv(EmptyRequest())

    def unpauseGazebo(self):
        self.unpause_srv(EmptyRequest())

    def rlServiceCallback(self,data):

        self.lowStateCallback(data.lowState)
        return self.sendCommandsToROS()


    #     _controlData->_legController->commands[i].kpCartesian = Vec3<double>(400,400,900).asDiagonal();
    # _controlData->_legController->commands[i].kdCartesian = Vec3<double>(20,20,20).asDiagonal();
    # _controlData->_legController->commands[i].pDes << 0, side_sign[i] * 0.083, 0;
    # _controlData->_legController->commands[i].pDes[2] = -h*progress + (1. - progress) * init_foot_pos(2, i);
    # // _data->_legController->commands[i].feedforwardForce = _data->_stateEstimator->getResult().rBody.transpose() * ff_force_world;
    def _test_latency_stand(self):
        """Just to test latency, from when the system is already standing, send standing commands. """
        # side_sign = [-1, 1, -1, 1];
        # legCmdArray = LegCmdArray()
        # for i in range(4):
        #     legCmdArray.legCmds[i].pDes = (0, side_sign[i]*0.083 , 0)#info['pDes'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].vDes = (0,0,0) #info['vDes'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].kpCartesianDiag = (400,400,900) #info['kpCartesianDiag'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].kdCartesianDiag = (30,30,30)  #info['kdCartesianDiag'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].feedforwardForce = (0,0,0) #info['feedforwardForce'][i*3:i*3+3]
        self._legCmd_pub.publish(self.legCmdArray)


    def receiveObsCallback(self,data):
        """Receive formatted obs from ROS, scale and step env, send back raw action """
        temptime = datetime.now() 
        obs = data.data

        if USE_RLLIB:
            normed_obs = self.venv.normalize_obs(obs)
            normed_obs = np.clip(normed_obs,
                                self.env.observation_space.low,
                                self.env.observation_space.high)
            #normed_obs = self.venv.normalize_obs(self._observation_contacts)
            action = self.model.compute_action(normed_obs)

        else:
            normed_obs = self.env.normalize_obs(obs)
            action = self.model.predict(normed_obs)[0]

        # scale action from [-1,1] to limits
        # rl_act = RLAct()
        self.rl_act.data = self._scale_helper(action, self._action_space_low, self._action_space_upp)

        self._rl_act_pub.publish(self.rl_act)
        print(datetime.now() - temptime)

    def lowStateCallback(self, data):
        """See LowState.msg for format. This is the full robot state in gazebo."""
        # print('*'*80)
        # print('lowStateCallback')
        # print(data)

        # pause sim while compute (should already be paused)
        #self.pauseGazebo()
        temptime = datetime.now() 
        #self._test_latency_stand()
        
        if 1:
            self.base_pos = data.cheat.position
            self.base_orn = np.array(data.cheat.orientation)
            # quaternion is in wrong order
            #self.base_orn = np.array([self.base_orn[1],self.base_orn[2],self.base_orn[3],self.base_orn[0]])
            self.base_orn = np.roll( self.base_orn,-1)
            # get base vels in world frame
            self.R = R = QuaternionToOrientationMatrix(self.base_orn)
            # in world frame
            self.base_linvel = R.T @ np.array(data.cheat.vBody)
            # not used, but this is in base frame
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
            self.feetInContactBool = np.where(np.array(data.footForce) > 0, 1, 0)
            self.hasStateData = True

            _observation_contacts = np.concatenate((self.joint_pos, 
                                                     self.joint_vel, 
                                                     self.joint_torques, 
                                                     self.base_orn,
                                                     [self.base_pos[2]],
                                                     self.feetInContactBool,
                                                     self.base_linvel))
            # self._observation_contacts = np.clip(_observation_contacts,
            #                                      self.env.observation_space.low,
            #                                      self.env.observation_space.high)
            self._observation_contacts = np.clip(_observation_contacts,
                                         self.env.observation_space.low,
                                         self.env.observation_space.high)

            self.sendCommandsToROS()
        print(datetime.now() - temptime)
        # print('low ', self.venv.observation_space.low)
        # print('high', self.venv.observation_space.high)
        # print('clipped',self._observation_contacts )
        # for i,j,k in zip(self.venv.observation_space.low, self.venv.observation_space.high, self._observation_contacts):
        #     print(i,j,k)
        # Note: need to vecnormalize these before stepping into env
        # if have venv as a vecnormalize obj, can do:
        # normed_obs = venv.normalize_obs(self._observation_contacts)
        # and then step:
        # compute_actions(normed_obs)

        # this may be taking up time, don't really need to do this
        # TODO: decouple general robotics functions from pybullet
        # self.quadruped_env._robot.SetFullState(self.base_pos,self.base_orn,
        #                             bodyLinVel=self.base_linvel,bodyAngVel=None,
        #                             jointPositions=self.joint_pos,jointVels=self.joint_vel)
        

    def sendCommandsToROS(self):
        """Send commands to ROS 

        Normalize actions with venv
        """
        if USE_RLLIB:

            # see how different these really are
            # print('Monitor total obs')
            # full_obs = self.env.reset()
            # print(full_obs)
            # print('NormAct total obs')
            # full_obs = self.env.env.reset()
            # print(full_obs)
            # print('Dummy   total obs')
            # full_obs = self.env.env.env.reset()
            # print(full_obs)
            # print('VecNorm total obs')
            # full_obs = self.env.env.env.env.reset()
            # print(full_obs)

            # print('quadruped_env')
            # reset_obs = self.quadruped_env.reset()
            # print(reset_obs)
            # print( self.quadruped_env.reset())
            #print( self.quadruped_env._robot.GetObservationContacts()) # this is set from ROS, so it's the same
            #print('ros obs') 
            #print( self._observation_contacts ) # this is before normalizing

            # print('obs_rms', self.venv.obs_rms)
            # print('obs_rms mean', self.venv.obs_rms.mean)
            # print('obs_rms var ', self.venv.obs_rms.var)
            # print('gamma', self.venv.gamma)
            # print('training', self.venv.training)
            # print('norm obs', self.venv.norm_obs)

            #for i,j,k,l in zip(self.venv.observation_space.low, self.venv.observation_space.high, reset_obs,self._observation_contacts,):
            #    print(np.array([i,j,k,l]), i<l and l<j)

            
            # just to see if this works
            # normed_obs = self.venv.normalize_obs(reset_obs)
            # print('norm quad obs')
            # print(normed_obs)
            # #action = self.model.compute_action(self.env.reset())
            # print(normed_obs.shape, self.env.reset().shape)

            # for i,j,k,l,m in zip(self.venv.observation_space.low, self.venv.observation_space.high, full_obs, reset_obs, normed_obs):
            #     print(np.array([i,j,k,l,m]))

            # there is some problem with 2 values - z height and in the orientation
            # THIS IS INDEPENDENT OF ANY ROS STUFF - even just using the rllib observation!!!??
            normed_obs = self.venv.normalize_obs(self._observation_contacts)
            normed_obs = np.clip(normed_obs,
                                self.env.observation_space.low,
                                self.env.observation_space.high)
            #normed_obs = self.venv.normalize_obs(self._observation_contacts)
            action = self.model.compute_action(normed_obs)

            # orig
            # normed_obs = self.venv.normalize_obs(self._observation_contacts)
            #action = self.model.compute_action(normed_obs)
            #tau, info = self.quadruped_env._robot.ScaleActionToImpedanceControl(action)
            # tau, info = self.ScaleActionToImpedanceControl(action)
            tau, legCmdArray = self.ScaleActionToImpedanceControl(action)


            # normed_obs = self.venv.normalize_obs(self._observation_contacts)
            # action = self.agent.compute_action(normed_obs)

        else:
            normed_obs = self.env.normalize_obs(self._observation_contacts)
            action = self.model.predict(normed_obs)
            #action = self.model.predict(self._observation_contacts)
            #print('action ', action[0])
            #tau, info = self.quadruped_env._robot.ScaleActionToImpedanceControl(action[0])
            #tau, info = self.ScaleActionToImpedanceControl(action[0])
            tau, legCmdArray = self.ScaleActionToImpedanceControl(action[0])

        # scale up action
        # need to scale the action according to ROS state, not pybullet state
        # can (1) set pyb state exactly equal to ROS first
        # (2) scale with 
        # will be easiest to set full state in pyb
        #data = LegCmdArray()
        #data.motorCmd = tau

        #data = RLServiceResponse()
        #data.legCmdArray.motorCmd = tau

        # legCmdArray = LegCmdArray()
        # legCmdArray.motorCmd = tau
        # for i in range(4):
        #     legCmdArray.legCmds[i].pDes = info['pDes'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].vDes = info['vDes'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].kpCartesianDiag = info['kpCartesianDiag'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].kdCartesianDiag = info['kdCartesianDiag'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].feedforwardForce = info['feedforwardForce'][i*3:i*3+3]

        #print(legCmdArray)

        # unpause sim
        #self.unpauseGazebo()
        self._legCmd_pub.publish(legCmdArray)
        #print('send', tau)
        #self.pauseGazebo()
        #return data


    def _scale_helper(self, action, lower_lim, upper_lim):
        """Helper to linearly scale from [-1,1] to lower/upper limits. 
        This needs to be made general in case action range is not [-1,1]
        """
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        # verify clip
        new_a = np.clip(new_a, lower_lim, upper_lim)
        return new_a

    def ScaleActionToImpedanceControl(self,actions,use_force_if_contact_only=True):
        """ Scale action form [-1,1] to everything needed for impedance control. 
        Assumes the following actions per control mode:
        IMPEDANCE:
          -4 * (x,y,z) pos; 4 x normal force (16 values)
        IMPEDANCE_FORCE:
          -4 * normal force 
        IMPEDANCE_GAINS:
          -4 * (x,y,z) pos; 4 x Fn; kpCartesian; kdCartesian (22 values)
        IMPEDANCE_3D_FORCE:
          -4 * (x,y,z) pos; 4 x 3D contact force (24 values)

        """
        # clip to make sure all actions in [-1,1]
        u = np.clip(actions,-1,1)

        # kp Cartesian and kdCartesian should in config file now
        kpCartesian = self.quadruped_env._robot_config.kpCartesian
        kdCartesian = self.quadruped_env._robot_config.kdCartesian

        if len(u) == 4: 
          # action is 4 normal forces, set des feet pos manually to "natural", just for testing
          # use feet positions
          hip_offset = self._robot_config.hip_offset #0.083
          zd_hip = 0.4
          Pd = np.array([0,-hip_offset,-zd_hip])
          u_pos = np.concatenate(FormatFRtoAll(Pd))
          u_force = u

        elif len(u) == 16: 
          # action is 12 positions + 4 normal forces
          u_pos = u[:12]
          u_force = u[12:]
          # scale des xyz 
          low_lim_ik = self._robot_config.IK_POS_LOW_LEG
          upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
          u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
          # scale des normal forces 
          low_lim_Fn = self._robot_config.MIN_NORMAL_FORCE
          upp_lim_Fn = self._robot_config.MAX_NORMAL_FORCE
          u_force = self._scale_helper(u_force, low_lim_Fn, upp_lim_Fn) 
          #u_force = [ np.array([0.,0.,fn]) for fn in u_force ]
          #u_force = np.concatenate( [R.T @ ffwd for ffwd in u_force])
          #u_force = np.concatenate( [R.T @ np.array([0.,0.,fn]) for fn in u_force])
          u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])

        elif len(u) == 22: 
          # action is 12 positions, 4 Fn, 3 kpCartesian, 3 kdCartesian
          u_pos = u[:12]
          u_force = u[12:16]
          kpCartesian = u[16:19]
          kdCartesian = u[19:]
          # scale des xyz 
          low_lim_ik = self._robot_config.IK_POS_LOW_LEG
          upp_lim_ik = self._robot_config.IK_POS_UPP_LEG
          u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
          # scale gains
          kpCartesian = np.diag(self._scale_helper(kpCartesian, 100, 800))
          kdCartesian = np.diag(self._scale_helper(kdCartesian,  10,  30))
          # scale des normal forces 
          low_lim_Fn = self._robot_config.MIN_NORMAL_FORCE
          upp_lim_Fn = self._robot_config.MAX_NORMAL_FORCE
          u_force = self._scale_helper(u_force, low_lim_Fn, upp_lim_Fn) 
          u_force = np.concatenate( [np.array([0.,0.,fn]) for fn in u_force])

        elif len(u) == 24: 
          # action is 12 positions, 4 3D contact forces
          u_pos = u[:12]
          u_force = u[12:]
          # scale des xyz 
          low_lim_ik = self.quadruped_env._robot_config.IK_POS_LOW_LEG
          upp_lim_ik = self.quadruped_env._robot_config.IK_POS_UPP_LEG
          u_pos = self._scale_helper(u_pos, low_lim_ik, upp_lim_ik) 
          # scale 3D contact force
          low_lim_F = self.quadruped_env._robot_config.MIN_NORMAL_FORCE_3D
          upp_lim_F = self.quadruped_env._robot_config.MAX_NORMAL_FORCE_3D
          u_force = self._scale_helper(u_force, low_lim_F, upp_lim_F) 

        else:
          raise ValueError('not implemented impedance control with this action space')

        # build kpCartesianArr, apply kpCartesian only if not in contact with ground, like in MPC
        kpCartesianArr = [kpCartesian]*4

        # adjust feedforward force only if in contact
        if use_force_if_contact_only:
          feetInContactBool = self.feetInContactBool
          for i,inContact in enumerate(feetInContactBool):
            if inContact == 0:
              #u_force[i] = 0
              u_force[3*i:3*i+3] = np.zeros(3)
            else:
              kpCartesianArr[i] = np.zeros((3,3))


        # desired foot velocity (may want to change)
        vd = np.zeros(3)

        # find corresponding torques
        action = np.zeros(12)
        # body pos, orientation matrix, velocity
        #basePos = np.asarray(self.GetBasePosition())
        R = self.R

        # save for ROS
        # feedforwardForce = np.zeros(12)
        # pDes = np.zeros(12)
        # vDes = np.zeros(12)
        # kpCartesianDiag = np.zeros(12)
        # kdCartesianDiag = np.zeros(12)
        legCmdArray = LegCmdArray()

        for i in range(4):
          # Jacobian and current foot position in hip frame
          J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(i)
          # desired foot position
          Pd = u_pos[i*3:i*3+3]
          # desired force (rotated into body frame)
          #Fd = R.T @ np.array([0.,0.,u_force[i]])
          Fd = R.T @ u_force[i*3:i*3+3]
          # foot velocity in leg frame
          foot_linvel = J @ self.joint_vel[i*3:i*3+3]#self.GetMotorVelocities()[i*3:i*3+3]
          # calculate torque
          #tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
          tau = J.T @ ( Fd + kpCartesianArr[i] @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
          action[i*3:i*3+3] = tau

          # save info
          # feedforwardForce[i*3:i*3+3] = Fd
          # pDes[i*3:i*3+3] = Pd
          # vDes[i*3:i*3+3] = vd
          # kpCartesianDiag[i*3:i*3+3] = kpCartesianArr[i].diagonal()
          # kdCartesianDiag[i*3:i*3+3] = kdCartesian.diagonal()
          legCmdArray.legCmds[i].pDes = tuple(Pd) #info['pDes'][i*3:i*3+3]
          legCmdArray.legCmds[i].vDes = tuple(vd) #info['vDes'][i*3:i*3+3]
          legCmdArray.legCmds[i].kpCartesianDiag = tuple(kpCartesianArr[i].diagonal()) #info['kpCartesianDiag'][i*3:i*3+3]
          legCmdArray.legCmds[i].kdCartesianDiag = tuple(kdCartesian.diagonal()) #info['kdCartesianDiag'][i*3:i*3+3]
          legCmdArray.legCmds[i].feedforwardForce = tuple(Fd) #info['feedforwardForce'][i*3:i*3+3]

        # for ROS
        # info = {}
        # info['feedforwardForce'] = feedforwardForce
        # info['pDes'] = pDes
        # info['vDes'] = vDes
        # info['kpCartesianDiag'] = kpCartesianDiag
        # info['kdCartesianDiag'] = kdCartesianDiag
        legCmdArray.motorCmd = tuple(action)
        # for i in range(4):
        #     legCmdArray.legCmds[i].pDes = info['pDes'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].vDes = info['vDes'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].kpCartesianDiag = info['kpCartesianDiag'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].kdCartesianDiag = info['kdCartesianDiag'][i*3:i*3+3]
        #     legCmdArray.legCmds[i].feedforwardForce = info['feedforwardForce'][i*3:i*3+3]


        return action, legCmdArray#info


    def _ComputeLegImpedanceControlActions(self,u_pos,u_vel,kpCartesian,kdCartesian,u_force,legID):
        """Get torques for specific forces and actions. 
        Note forces coming from ROS are going to be Vec3 that are already rotated in local frame.
        This function only for 1 leg
        """
        # rotation matrix
        #R = self.GetBaseOrientationMatrix()
        # Jacobian and current foot position in hip frame
        J, ee_pos_legFrame = self._ComputeJacobianAndPositionUSC(legID)
        # desired foot position
        Pd = u_pos 
        # desired foot velocity 
        vd = u_vel 
        # desired force (rotated)
        Fd = u_force 
        # foot velocity in leg frame
        foot_linvel = J @ self.joint_vel[legID*3:legID*3+3]#self.GetMotorVelocities()[legID*3:legID*3+3]
        # calculate torque
        tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
        return tau

    def _ComputeJacobianAndPositionUSC(self, legID):
        """Translated to python/pybullet from: 
        https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/LegController.cpp
        can also use pybullet function calculateJacobian()
        IN LEG FRAME
        (see examples in Aliengo/pybullet_tests/aliengo_jacobian_tests.py)
        # from LegController:
        # * Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
        """

        # joint positions of leg
        #q = self.GetMotorAngles()[legID*3:legID*3+3]
        q = self.joint_pos[legID*3:legID*3+3]
        # Foot position in base frame
        pos = np.zeros(3)

        l1 = self.quadruped_env._robot_config.HIP_LINK_LENGTH
        l2 = self.quadruped_env._robot_config.THIGH_LINK_LENGTH
        l3 = self.quadruped_env._robot_config.CALF_LINK_LENGTH

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

    def stepFromMPC(self, data, calcTorque=True):
        """Take data from LegCmdArray and step simulation (torque control mode). """
        # torques calculated from ROS
        tau_actions = np.array(data.motorCmd)

        all_Fns = np.zeros(12)
        all_pDes = np.zeros(12)

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
                    vDes = np.zeros(3)
                    all_Fns[i*3:i*3+3] = u_force
                    all_pDes[i*3:i*3+3] = pDes
                    actions[i*3:i*3+3] = self.env._robot._ComputeLegImpedanceControlActions(pDes,
                                                                                            vDes,
                                                                                            kpCartesian,
                                                                                            kdCartesian,
                                                                                            u_force,
                                                                                            i #legID
                                                                                            )
                self.env.step(actions)
            else:
                self.env.step(tau_actions)  
                break
        
        # compare calculated torques and those from ROS
        #print('*'*50)
        #nicePrint(all_Fns)
        #print(self.env._robot.GetBaseLinearVelocity())
        #self.env.step(tau_actions)  
        #self.env.step(actions) 


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
        # step sim
        self.stepFromMPC(leg_cmds.legCmdArray)

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
    "isRLGymInterface": False, 
    "robot_config": robot_config, 
    "time_step": 0.001, # make sure this is small enough
    "render": False, 
    "motor_control_mode": "TORQUE"
}

#_rosBridge = QuadrupedRLROSInterface(env_config)

# _rosBridge = QuadrupedRLROSInterface(rl_env, agent)
_rosBridge = QuadrupedRLROSInterface(env, model)
# for _ in range(1000):
#     _rosBridge.sendAvailable()
#     _rosBridge._rate.sleep()

time.sleep(1)
counter = 0
while not rospy.is_shutdown():
    #_rosBridge._rate.sleep()
    counter+= 1
    #_rosBridge.sendAvailable()

#while True:
# send robot state to ROS
#_rosBridge.getAndSetActionFromMPC()
# if fallen, reset and choose random new gait
# if counter > 10000 or abs(_rosBridge.env._robot.GetBaseOrientationRollPitchYaw()[0]) > 1: 
#     _rosBridge.resetMPC()
#     _rosBridge.setGait(int(np.random.randint(6)+1))
#     _rosBridge.env.reset()
#     counter = 0

#_rosBridge._rate.sleep()
#counter += 1
if USE_RLLIB:
    ray.shutdown()