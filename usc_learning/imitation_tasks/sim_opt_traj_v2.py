import os
import pybullet as p
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt

from usc_learning.imitation_tasks.traj_motion_data import TrajTaskMotionData
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
# import configs
import usc_learning.envs.quadruped_master.configs_a1 as robot_config
# check aliengo first
#import usc_learning.envs.quadruped_master.configs_aliengo as robot_config
import usc_learning


from usc_learning.utils.utils import nicePrint, nicePrint2D

""" Simulate optimal trajectory. Refactor to use general Quadruped env. """
ROBOT = 1
# 0 is Aliengo
# 1 is A1
SIM_FLAG = False     # sim joint PD control
SIM_IMPEDANCE = True # sim impedance control 
control_mode = "TORQUE"
time_step = 0.001
action_repeat = 1

def get_opt_trajs(path):
    """ Returns all optimal trajectories in directory. """
    # try:
    #   files =  os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), path)
    # except:
    path = os.path.join(os.path.dirname(os.path.abspath(usc_learning.imitation_tasks.__file__)),path)
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if (basename.endswith('.csv') and 'cartesian' not in basename)]
    return paths 

opt_data = "data/a1_v0.csv"
opt_data = "data/jumpingFull_A1.csv"
opt_data = 'data/jumpingFull_A1_1ms_h50_d60.csv'

opt_data = 'data2/jumpingFull_A1_1ms_h50_d60.csv'


opt_data = 'data4/jumpingFull_A1_1ms_h40_d60.csv'

opt_trajs = get_opt_trajs('data4')
traj_idx = np.random.randint(len(opt_trajs))
opt_data = opt_trajs[traj_idx]
print('using traj', opt_data)
# backflip
# opt_data = '~/home/guillaume/Documents/MATLAB/Quadruped-Robots-Optimization/jumping_up/trajs/backflipFull_A1_1ms_h0_d-50.csv'
# opt_data = 'data3/backflipFull_A1_1ms_h0_d-50.csv'
# if ROBOT == 0:
#     opt_data = 'data/jumpingFull.csv'
#     opt_data = 'data/jumpingFull_1ms.csv'

# check some data in data4


robot_config.MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 1

env = QuadrupedGymEnv(
      robot_config=robot_config,
      isRLGymInterface=False, # True means running RL, should be false if running tests
      action_repeat=action_repeat,
      time_step=time_step,
      accurate_motor_model_enabled=True,
      motor_control_mode=control_mode,
      hard_reset=True,
      # record_video=True,
      render=True)

#robot = env._robot
# set ground coefficient of friction
#env._pybullet_client.changeDynamics(env.plane, -1, lateralFriction=0.5)
# reset plane orientation
orn_noise = 0.1 
# plane_orn = orn_noise/2*np.array([np.random.random(),np.random.random(),0]) - orn_noise
# plane_orn = orn_noise/2*np.array([1,1,0]) - orn_noise
# plane_orn = env._pybullet_client.getQuaternionFromEuler(plane_orn)
# env._pybullet_client.resetBasePositionAndOrientation(env.plane, 
#                                             [0,0,0],
#                                             plane_orn)

# env.add_box_ff(0.03)
# env.add_box_rr(0.06)
# env._pybullet_client.changeDynamics(env.box_ff, -1, lateralFriction=1)
# env._pybullet_client.changeDynamics(env.box_rr, -1, lateralFriction=1)

if ROBOT == 0 and opt_data == 'data/jumpingFull.csv':
    traj_task = TrajTaskMotionData(filename=opt_data,dt=0.01)
else:
    traj_task = TrajTaskMotionData(filename=opt_data,dt=0.001)

if SIM_IMPEDANCE:
    traj_task = TrajTaskMotionData(filename=opt_data,dt=0.001,useCartesianData=True)


def get_traj_state_at_index(idx):
    """ Get the traj state at time t. 
    Return:
    -base_pos
    -base_orn
    -base_linvel
    -base_angvel
    -joint_pos
    -joint_vel
    """
    traj_state = traj_task.get_state_at_time(idx)
    base_pos = traj_task.get_base_pos_from_state(traj_state)#,env2D=ENV_2D_FLAG)
    base_orn = traj_task.get_base_orn_from_state(traj_state)#,env2D=ENV_2D_FLAG)
    base_linvel = traj_task.get_base_linvel_from_state(traj_state)#,env2D=ENV_2D_FLAG)
    base_angvel = traj_task.get_base_angvel_from_state(traj_state)#,env2D=ENV_2D_FLAG)

    joint_pos = traj_task.get_joint_pos_from_state(traj_state)#,env2D=ENV_2D_FLAG)
    joint_vel = traj_task.get_joint_vel_from_state(traj_state)#,env2D=ENV_2D_FLAG)
    
    return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

def reset_robot():
    """ Reset robot to beginning position. """
    #robot.Reset()
    #env.reset()
    #env.add_box_at(0.5,0.6)
    base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel = get_traj_state_at_index(0)

    #base_pos[2] += 0.1

    init_base = np.concatenate((base_pos, base_orn))
    init_joints = np.concatenate((joint_pos, joint_vel))
    env._robot.SetPoseAndSettle(init_base, init_joints)
    #robot._disable_motor()
    print('just done with settle')
    """add box """
    # env.add_box()
    

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    #print(vector)
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def convertAnglesToPlusMinusPi2(angles, _last_base_euler):
    """Converts angles to +/- pi/2 """
    # new_angles = []
    # for a in angles:
    #   while a > np.pi/2:
    #     a -= np.pi
    #   while a < -np.pi/2:
    #     a += np.pi
    #   new_angles.append(a)
    #new_angles = np.zeros(len(angles))
    base_orn = env._robot.GetBaseOrientation()
    new_angles = angles.copy()
    for i, a in zip([0, 2, 1], [angles[0], angles[2], angles[1]]): #enumerate(angles):
        #print(i,a)
        if i ==1:
            # check if the value of transforming pitch back to quaternion space is same
            # if not, check value, and can subtract it from pi or -pi
            #print('current quaternion', base_orn)
            #print('pitch from conversion', a)
            new_base_orn = env._pybullet_client.getQuaternionFromEuler([new_angles[0], a, new_angles[2]])
            #print('pitch to quaternion', env._pybullet_client.getQuaternionFromEuler([angles[0], a, angles[2]]))
            if not np.allclose(base_orn, new_base_orn, rtol=1e-3, atol=1e-3):
            # if np.linalg.norm(np.array(base_orn) - np.array(new_base_orn)) > 5e-7:
            #if abs(base_orn[1] - new_base_orn[1]) > 1e-4:
                if a <= 0:
                    new_angles[i] = -np.pi - a
                else: # should never get here
                    new_angles[i] = np.pi - a
            else:
                new_angles[i] = a

        elif abs(_last_base_euler[i] - a) > np.pi/2:
            #print('i', i)
            if _last_base_euler[i] > new_angles[i]:
                while _last_base_euler[i] > new_angles[i] and abs(_last_base_euler[i] - new_angles[i]) > np.pi/2:
                    # skipped across 0 to around pi
                    # print('pre  update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
                    new_angles[i] += np.pi
                    # print('post update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
            elif _last_base_euler[i] < new_angles[i]:
                while _last_base_euler[i] < new_angles[i] and abs(_last_base_euler[i] - new_angles[i]) > np.pi/2:
                    # print('pre  update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
                    new_angles[i] -= np.pi
                    # print('post update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
        else:
            new_angles[i] = a
    return np.array(new_angles)


class pdTrajController():
    def __init__(self,Kp=None,Kd=None):
        self.setGainsDefault(Kp=Kp,Kd=Kd)

    def setGainsDefault(self,Kp=None,Kd=None):
        if Kp == None:
            # taken from https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/body.cpp
            """
            TODO: tune below gains
            """
            self.kps = np.asarray([70,180,300]*4)
            self.kds = np.asarray([3,8,15] * 4)
            self.kps = np.asarray([70,300,300]*4)
            self.kds = np.asarray([3,20,20] * 4)
            #self.kps = np.asarray([70,1000,300]*2 + [70,300,300]*2)#BEST
            #self.kds = np.asarray([3,20,80]*2 + [3,15,12]*2)#BEST

            # from gym env
        else:
            self.kps = np.asarray([Kp]*12)
            self.kds = np.asarray([Kd]*12)
        if ROBOT == 1: #A1
            # these worked
            self.kps = np.asarray([300,500,500]*4)
            self.kds = np.asarray([2,3,3] * 4)
            # experimental, overshoot dist, under height
            self.kps = np.asarray([300,200,400]*4)
            self.kds = np.asarray([10,5,5] * 4)
            # this works well for 0.5m height, 0.6m distance
            self.kps = np.asarray([400,500,500]*4)
            self.kds = np.asarray([7,5,5] * 4)

            # self.kps = np.asarray([300,300,300]*4)
            # self.kds = np.asarray([3,3,3] * 4)

            # self.kps = np.asarray([400,400,400]*4)
            # self.kds = np.asarray([7,5,5] * 4) # 4 good here as well

            # self.kps = np.asarray([200,200,200]*4)
            # self.kds = np.asarray([2,2,2] * 4) 

            # # # not good
            # self.kps = np.asarray([50,50,50]*4)
            # self.kds = np.asarray([1,1,1] * 4) 

            self.kps = np.asarray([100,100,100]*4)
            self.kds = np.asarray([2,2,2] * 4) 

            self.kps = np.asarray([300,300,300]*4)
            self.kds = np.asarray([3,3,3] * 4)

            # self.kps = np.asarray([400,400,400]*4)
            # self.kds = np.asarray([4,4,4] * 4)

            # self.kps = np.asarray([0]*12)
            # self.kds = np.asarray([0]*12) 

            # not very food
            # self.kps = np.asarray([500,500,500]*4)
            # self.kds = np.asarray([5,5,5] * 4)

    def calcTorques(self, q_des, q, dq_des, dq, tau):
        return self.kps*(q_des-q) + self.kds*(dq_des-dq) - tau

    def setGains(self, kp, kd):
        """Set kp, kd """
        self.kps = kp
        self.kds = kd

if ROBOT == 0: # ALiengo
    jc = pdTrajController()#300,5)#1,0.2)
else:
    jc = pdTrajController()#0,0)

# gather trajectory to print later
body_x = []
body_z = []
body_pitch = []

des_body_x = []
des_body_z = []
des_body_pitch = []

des_foot1 = np.zeros((3,traj_task.trajLen))
act_foot1 = np.zeros((3,traj_task.trajLen))

#while True:
for traj in opt_trajs:
    print("\n reset \n")
    print("*"*100)
    print('traj', traj )
    opt_data = traj

    if ROBOT == 0 and opt_data == 'data/jumpingFull.csv':
        traj_task = TrajTaskMotionData(filename=opt_data,dt=0.01)
    else:
        traj_task = TrajTaskMotionData(filename=opt_data,dt=0.001)

    if SIM_IMPEDANCE:
        traj_task = TrajTaskMotionData(filename=opt_data,dt=0.001,useCartesianData=True)

    
    reset_robot()
    for _ in range(1000):
        i = 0
        des_state = traj_task.get_state_at_index(i)
        des_torques = traj_task.get_torques_at_index(i)
        q_des = traj_task.get_joint_pos_from_state(des_state)
        dq_des = traj_task.get_joint_vel_from_state(des_state)
        q = env._robot.GetMotorAngles()
        dq = env._robot.GetMotorVelocities()
        # this should be 0 to settle
        des_torques = np.zeros(12)
        jc.setGains(np.array([220]*12), np.array([2]*12))
        u = jc.calcTorques(q_des, q, dq_des, dq, des_torques)
        u = np.clip(u,
                    -robot_config.TORQUE_LIMITS,
                    robot_config.TORQUE_LIMITS)

        #u = [ 0.000,  1.257, -2.356] * 4
        #u[0] = 0.2
        #robot._control_mode = "PD"
        #robot.Step(u)
        env.step(u)

        #robot._control_mode = "TORQUE"
        
        # print('init torques', max(u))
        # print('q')
        # nicePrint(q)
        # print('qdes')
        # nicePrint(q_des)
        # print('des torques')
        # nicePrint(des_torques)
        #time.sleep(0.01)

    _last_base_euler = env._robot.GetBaseOrientationRollPitchYaw()
    jc.setGainsDefault()
    #while True:
    for i in range(traj_task.trajLen): #  in range(800): #
    #for asdfasdf in range(5):
        #i=0
        #print("-"*50)
        #print(i)
        des_state = traj_task.get_state_at_index(i)
        des_torques = traj_task.get_torques_at_index(i)
        q_des = traj_task.get_joint_pos_from_state(des_state)
        dq_des = traj_task.get_joint_vel_from_state(des_state)

        curr_base_pos = env._robot.GetBasePosition()
        curr_base_orn = env._robot.GetBaseOrientationRollPitchYaw()
        curr_base_orn = convertAnglesToPlusMinusPi2(curr_base_orn, _last_base_euler)
        body_x.append(curr_base_pos[0])
        body_z.append(curr_base_pos[2])
        body_pitch.append(curr_base_orn[1])
        des_body_x.append(des_state[0])
        des_body_z.append(des_state[2])
        des_body_pitch.append(des_state[4])

        _last_base_euler = curr_base_orn

        # for j in range(4):
        #   print(j, env._robot.GetLocalEEPos(j))
        #   print(env._robot._ComputeJacobianAndPositionUSC(j)[1]) # ee leg pos in leg frame

        if SIM_FLAG:
            #for _j in range(10):
            q = env._robot.GetMotorAngles()
            dq = env._robot.GetMotorVelocities()
            print('q')
            nicePrint(q)
            print('qdes')
            nicePrint(q_des)
            print(des_torques)
            u = jc.calcTorques(q_des, q, dq_des, dq, des_torques)
            u = np.clip(u,
                -robot_config.TORQUE_LIMITS,
                robot_config.TORQUE_LIMITS)
            #u = np.clip(u,-55,55)
            #u=des_torques
            #robot.Step(u)
            print("u")
            nicePrint(u)
            env.step(u) # since in PD control
            time.sleep(0.005)

        elif SIM_IMPEDANCE:
            # get feedforward torques from joint PD control
            q = env._robot.GetMotorAngles()
            dq = env._robot.GetMotorVelocities()
            u = jc.calcTorques(q_des, q, dq_des, dq, des_torques)
            u = np.clip(u,
                -robot_config.TORQUE_LIMITS,
                robot_config.TORQUE_LIMITS)

            # u = np.zeros(12)
            # u= - des_torques
            # check impedance control
            foot_pos = traj_task.get_foot_pos_from_state(des_state)
            foot_vel = traj_task.get_foot_vel_from_state(des_state)
            foot_force = traj_task.get_foot_force_from_state(des_state)

            kpCartesian = np.diag([500,500,500])
            kdCartesian = np.diag([10,10,10])

            # kpCartesian = np.diag([800,800,800])
            # kdCartesian = np.diag([20,20,20])
            # kpCartesian = np.diag([100,100,100])
            # kdCartesian = np.diag([20,20,20])
            # kpCartesian = np.diag([800,800,800])
            # kdCartesian = np.diag([20,20,30])
            # # ok but backwards
            # # kpCartesian = np.diag([2100,2100,2100])
            # # kdCartesian = np.diag([100,100,100])
            # # tests
            # kpCartesian = np.diag([5000,20000,5000])
            # kdCartesian = np.diag([300,600,300])

            # kpCartesian = np.diag([800,800,800])
            # kdCartesian = np.diag([20,20,30])
            # kpCartesian = np.diag([0,0,0])
            # kdCartesian = np.diag([0,0,0])

            # kpCartesian = np.diag([800,800,800])
            # kdCartesian = np.diag([30,30,30])

            actions = np.zeros(12)

            zero_force = True

            #print('----------------------------------------------')
            for j in range(4):

                if zero_force:
                    u_force = np.zeros(3)
                else:
                    u_force = foot_force[j*3:j*3+3]
                #print('des_pos ', foot_pos[j*3:j*3+3])
                J, ee_pos_legFrame = env._robot._ComputeJacobianAndPositionUSC(j)
                #print('act_pos ', ee_pos_legFrame )
                if j == 0:
                    des_foot1[:,i] = foot_pos[j*3:j*3+3]
                    act_foot1[:,i] = ee_pos_legFrame
                actions[j*3:j*3+3] = env._robot._ComputeLegImpedanceControlActions(foot_pos[j*3:j*3+3],
                                                                                    foot_vel[j*3:j*3+3],
                                                                                    kpCartesian,
                                                                                    kdCartesian,
                                                                                    u_force,
                                                                                    j)
            env.step(actions+u)
            #env.step(actions-des_torques)
            time.sleep(0.0001)

        else:
            base_pos = traj_task.get_base_pos_from_state(des_state)#,env2D=ENV_2D_FLAG)
            base_orn = traj_task.get_base_orn_from_state(des_state)#,env2D=ENV_2D_FLAG)
            base_linvel = traj_task.get_base_linvel_from_state(des_state)#,env2D=ENV_2D_FLAG)
            base_angvel = traj_task.get_base_angvel_from_state(des_state)#,env2D=ENV_2D_FLAG)
            #robot.SetFullState(base_pos,base_orn,base_linvel,base_angvel,q_des,dq_des)
            #robot.SetFullState(base_pos,base_orn,jointPositions=q_des)
            # bodyAngVel=base_angvel
            # bodyLinVel=base_linvel
            env._robot.SetFullState(base_pos,base_orn,bodyLinVel=base_linvel,
                                                bodyAngVel=base_angvel,
                                                jointPositions=q_des,
                                                jointVels=dq_des)


    time.sleep(.5)
    #break



print('final base pos', env._robot.GetBasePosition())
# plt.plot(body_x,body_z, 'b-',label='actual')
# plt.plot(traj_task.baseXYZ[0,:], traj_task.baseXYZ[2,:], 'r-',label='desired')
plt.plot(body_x,label='actual x')
plt.plot(body_z,label='actual z')
plt.plot(des_body_x, label='des x')
plt.plot(des_body_z, label='des z')
plt.legend()
plt.show()
#plt.clf()

plt.figure()
# plt.plot(traj_task.baseRPY[1,:], label='des pitch')
# plt.plot(body_pitch, label='true pitch')
plt.plot(body_pitch, label='actual pitch')
plt.plot(des_body_pitch, label='des pitch')
plt.legend()
plt.show()

plt.figure()
plt.plot(des_foot1[0,:], label='des x')
plt.plot(des_foot1[1,:], label='des y')
plt.plot(des_foot1[2,:], label='des z')
plt.plot(act_foot1[0,:], label='act x')
plt.plot(act_foot1[1,:], label='act y')
plt.plot(act_foot1[2,:], label='act z')
plt.legend()
plt.show()


