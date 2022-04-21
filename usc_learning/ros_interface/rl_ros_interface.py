"""Example file, self-contained.

Get robot state from ROS, get action from policy network, send back leg commands to ROS.

Need 3 terminals, with in order:

Terminal 1 (after source ROS path):
$ roslaunch laikago_gazebo aliengo.launch
Terminal 2 (after source ROS path):
$ rosrun laikago_gazebo pyb_mpc
Terminal 3 [this one] (source ROS path, activate virtualenv):
$ python rl_ros_interface.py


This is going to get into a mess, how best to merge this with rllib interface?

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
from std_msgs.msg import Empty, Int32
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

USE_RL = True
########################################################################################################
# Load RL stuff... this needs to be in its own file/class
########################################################################################################
USING_VEC_ENV = True
alg = "PPO"

ray_path = "/home/guillaume/Documents/Research/DRL/Reinforcement-Learning-for-Quadruped-Robots/usc_learning/learning/rllib"
log_dir = os.path.join(ray_path, 'ray_results','071220024610')

########################################
alg_dir = get_latest_directory(log_dir) #i.e. {log_dir}/PPO
train_dir = get_latest_directory(alg_dir) # i.e. {log_dir}/PPO/PPO_DummyQuadrupedEnv_0_2020-07-11_13-27-26rtt2kwpd

# most recent checkpoint in train_dir
checkpoint = get_latest_model_rllib(train_dir)
# visualize episode lengths/rewards over training
#data = load_rllib(train_dir)

# for loading env configs, vec normalize stats
stats_path = str(log_dir)
vec_stats_path = os.path.join(stats_path,"vec_normalize.pkl")

# get env config if available
try:
    # stats path has all saved files (quadruped, configs_a1, as well as vec params)
    env_config = read_env_config(stats_path)
    sys.path.append(stats_path)

    # check env_configs.txt file to see whether to load aliengo or a1
    with open(stats_path + '/env_configs.txt') as configs_txt:
        if 'aliengo' in configs_txt.read():
            import configs_aliengo as robot_config
        else:
            import configs_a1 as robot_config

    env_config['robot_config'] = robot_config
except:
    print('*'*50,'could not load env stats','*'*50)
    env_config = {}
    sys.exit()
env_config['render'] = True

class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, dummy_env_config):
        # set up like stable baselines (will this work?)
        #env = QuadrupedGymEnv(render=False,hard_reset=True)
        if USING_VEC_ENV:
            env = lambda: QuadrupedGymEnv(**env_config) #render=True,hard_reset=True
            env = make_vec_env(env, n_envs=1)
            env = VecNormalize.load(vec_stats_path, env)
            # do not update stats at test time
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False

            self.env = env 
            self._save_vec_stats_counter = 0
        else:
            self.env = QuadrupedGymEnv(**env_config)#render=True,hard_reset=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('\n','*'*80)
        print('obs high',self.observation_space.high)
        print('obs low', self.observation_space.low)

    def reset(self):
        """reset env """
        obs = self.env.reset()
        if USING_VEC_ENV:
            obs = obs[0]
            # no need to save anything at load time
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        """step env """
        if USING_VEC_ENV:
            obs, rew, done, info = self.env.step([action])
            obs, rew, done, info = obs[0], rew[0], done[0], info[0]
        else:
            obs, rew, done, info = self.env.step(action)
        #print('step obs, rew, done, info', obs, rew, done, info)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            sys.exit()
        return obs, rew, done, info 

# load training configurations (can be seen in params.json)
config_path = os.path.join(train_dir, "params.pkl")

with open(config_path, "rb") as f:
    config = pickle.load(f)
# complains if 1.. but is should be 1?
config["num_workers"] = 0

ray.init()
agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)
# Load state from checkpoint.
agent.restore(checkpoint)
# env to pass to ROS interface class
rl_env = agent.workers.local_worker().env
########################################################################################################
# RL Interface, this also needs to be in its own file/class
# Could merge DummyQuadrupedEnv with the below into
# ROSEnv, which contains the VecEnv/normal quadruped env, and all the communication related stuff
########################################################################################################
class QuadrupedRLROSInterface(object):
    """ Quadruped ROS MPC Interface """
    def __init__(self, env_opt, agent=None):
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
            # a vecnormalized quadruped gym env
            self.env = env_opt
            self.agent = agent
            self.venv = self.env.env.env.env
            self.quadruped_env = self.env.env.env.env.venv.envs[0].env #lol

        # print(self.env)                                 # rllib Monitor
        # print(self.env.env)                             # rllib NormalizeActionWrapper
        # print(self.env.env.env)                         # Dummy Quadruped Instance
        # print(self.env.env.env.env)                     # stable_baselines.common.vec_env.vec_normalize.VecNormalize
        # print(self.env.env.env.env.venv)                # dummy vec env
        # print(self.env.env.env.env.venv.envs[0].env)    # Quadruped Gym Env

        # sys.exit()

        # subscribers
        self._state_sub = rospy.Subscriber("/laikago_gazebo/lowState/state",LowState, self.lowStateCallback )
        # publish - legCmdArray
        self._legCmd_pub = rospy.Publisher("/laikago_gazebo/rl_actions",LegCmdArray, queue_size=1 )

        # # service client to call MPC with pyb state, and return leg commands
        # self._rate = rospy.Rate(1000) # careful
        # rospy.wait_for_service('/laikago_gazebo/mpc_comm')
        # try:
        #     self.mpc_comm = rospy.ServiceProxy('/laikago_gazebo/mpc_comm', MPCService,persistent=True)
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)

        # # publishers - reset MPC, and set gait
        # self._resetMPC_pub = rospy.Publisher("/laikago_gazebo/reset_mpc", Empty, queue_size=1)
        # self._setGait_pub = rospy.Publisher("/laikago_gazebo/set_gait", Int32, queue_size=1)



    def lowStateCallback(self, data):
        """See LowState.msg for format. This is the full robot state in gazebo."""
        #print('*'*80)
        #print('lowStateCallback')
        #print(data)
        self.base_pos = data.cheat.position
        self.base_orn = np.array(data.cheat.orientation)
        # quaternion is in wrong order
        #self.base_orn = np.array([self.base_orn[1],self.base_orn[2],self.base_orn[3],self.base_orn[0]])
        self.base_orn = np.roll( self.base_orn,-1)
        # get base vels in world frame
        R = QuaternionToOrientationMatrix(self.base_orn)
        # print('base orn', self.base_orn)
        # print('R', R)
        # print('vBody', data.cheat.vBody)
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
                                     self.venv.observation_space.low,
                                     self.venv.observation_space.high)
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
        self.quadruped_env._robot.SetFullState(self.base_pos,self.base_orn,
                                    bodyLinVel=self.base_linvel,bodyAngVel=None,
                                    jointPositions=self.joint_pos,jointVels=self.joint_vel)
        self.sendCommandsToROS()

    def sendCommandsToROS(self):
        """Send commands to ROS 

        Normalize actions with venv
        """
        # normed_obs = self.venv.normalize_obs(self._observation_contacts)
        # action = self.agent.compute_action(normed_obs)
        # tau = self.env._robot.ScaleActionToImpedanceControl(action)

        normed_obs = self.venv.normalize_obs(self._observation_contacts)
        action = self.agent.compute_action(normed_obs)
        tau, info = self.quadruped_env._robot.ScaleActionToImpedanceControl(action)

        # scale up action
        # need to scale the action according to ROS state, not pybullet state
        # can (1) set pyb state exactly equal to ROS first
        # (2) scale with 
        # will be easiest to set full state in pyb
        data = LegCmdArray()
        data.motorCmd = tau
        for i in range(4):
            data.legCmdArray.legCmds[i].pDes = info['pDes'][i*3:i*3+3]
            data.legCmdArray.legCmds[i].vDes = info['vDes'][i*3:i*3+3]
            data.legCmdArray.legCmds[i].kpCartesianDiag = info['kpCartesianDiag'][i*3:i*3+3]
            data.legCmdArray.legCmds[i].kdCartesianDiag = info['kdCartesianDiag'][i*3:i*3+3]
            data.legCmdArray.legCmds[i].feedforwardForce = info['feedforwardForce'][i*3:i*3+3]

        self._legCmd_pub.publish(data)



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
    "render": True, 
    "motor_control_mode": "TORQUE"
}

#_rosBridge = QuadrupedRLROSInterface(env_config)

_rosBridge = QuadrupedRLROSInterface(rl_env, agent)

time.sleep(1)
counter = 0
while not rospy.is_shutdown():
    #_rosBridge._rate.sleep()
    counter+= 1

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