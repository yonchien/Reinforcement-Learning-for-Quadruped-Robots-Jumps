import pybullet as p
import pybullet_utils.bullet_client as bullet_client
import numpy as np

from usc_learning.envs.aliengo import AlienGo 
from usc_learning.envs.aliengo2D import AlienGo2D
from usc_learning.imitation_tasks.traj_motion_data import TrajTaskMotionData

from usc_learning.envs.aliengo import TORQUE_LIMITS
from usc_learning.utils.utils import nicePrint, nicePrint2D

""" Simulate optimal trajectory. """
ENV_2D_FLAG = False
SIM_FLAG = True
control_mode = "TORQUE"
time_step = 0.001
action_repeat = 1

# p.connect(p.GUI) #
_pybullet_client =  bullet_client.BulletClient(connection_mode=p.GUI) 

if ENV_2D_FLAG:
	robot = AlienGo2D(pybullet_client=_pybullet_client,
									isGymInterface=False,
									control_mode=control_mode,
									time_step=time_step,
									actionRepeat=action_repeat)
else:
	robot = AlienGo(pybullet_client=_pybullet_client,
									isGymInterface=False,
									control_mode=control_mode,
									time_step=time_step,
									actionRepeat=action_repeat)

traj_task = TrajTaskMotionData()


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
	robot.Reset()
	base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel = get_traj_state_at_index(0)
	init_base = np.concatenate((base_pos,base_orn))
	init_joints = np.concatenate((joint_pos,joint_vel))
	robot.SetPoseAndSettle(init_base,init_joints)
	robot._disable_motor()


class pdTrajController():
	def __init__(self,Kp=None,Kd=None):
		if Kp == None:
			# taken from https://github.com/DRCL-USC/USC-AlienGo-Software/blob/master/laikago_ros/laikago_gazebo/src/body.cpp
			"""
			TODO: tune below gains
			"""
			self.kps = np.asarray([70,180,300]*4)
			self.kds = np.asarray([3,8,15] * 4)
			self.kps = np.asarray([70,1000,300]*2 + [70,300,300]*2)#BEST
			self.kds = np.asarray([3,20,80]*2 + [3,15,12]*2)#BEST
		else:
			self.kps = np.asarray([Kp]*12)
			self.kds = np.asarray([Kd]*12)
	def calcTorques(self, q_des, q, dq_des, dq, tau):
		return self.kps*(q_des-q) + self.kds*(dq_des-dq) #+ tau

jc = pdTrajController()#1,0.2)

while True:
	print("\n reset \n")
	reset_robot()
	for _ in range(200):
		i = 0
		des_state = traj_task.get_state_at_index(i)
		des_torques = traj_task.get_torques_at_index(i)
		q_des = traj_task.get_joint_pos_from_state(des_state)
		dq_des = traj_task.get_joint_vel_from_state(des_state)
		q = robot.GetJointAngles()
		dq = robot.GetJointVelocities()
		u = jc.calcTorques(q_des, q, dq_des, dq, des_torques)
		u = np.clip(u,-55,55)

		#u = [ 0.000,  1.257, -2.356] * 4
		#robot._control_mode = "PD"
		robot.Step(u)
		#robot._control_mode = "TORQUE"
		
		print('init torques', max(u))
		print('q')
		nicePrint(q)
		print('qdes')
		nicePrint(q_des)
		print('des torques')
		nicePrint(des_torques)

	for i in range(traj_task.trajLen):
		print(i)
		des_state = traj_task.get_state_at_index(i)
		des_torques = traj_task.get_torques_at_index(i)
		q_des = traj_task.get_joint_pos_from_state(des_state)
		dq_des = traj_task.get_joint_vel_from_state(des_state)

		if SIM_FLAG:
			for _j in range(10):
				q = robot.GetJointAngles()
				dq = robot.GetJointVelocities()
				# print('q')
				# nicePrint(q)
				# print('qdes')
				# nicePrint(q_des)
				u = jc.calcTorques(q_des, q, dq_des, dq, des_torques)
				u = np.clip(u,-55,55)
				#u=des_torques
				robot.Step(u)

		else:
			base_pos = traj_task.get_base_pos_from_state(des_state)#,env2D=ENV_2D_FLAG)
			base_orn = traj_task.get_base_orn_from_state(des_state)#,env2D=ENV_2D_FLAG)
			base_linvel = traj_task.get_base_linvel_from_state(des_state)#,env2D=ENV_2D_FLAG)
			base_angvel = traj_task.get_base_angvel_from_state(des_state)#,env2D=ENV_2D_FLAG)
			#robot.SetFullState(base_pos,base_orn,base_linvel,base_angvel,q_des,dq_des)
			#robot.SetFullState(base_pos,base_orn,jointPositions=q_des)
			# bodyAngVel=base_angvel
			# bodyLinVel=base_linvel
			robot.SetFullState(base_pos,base_orn,bodyLinVel=base_linvel,
												bodyAngVel=base_angvel,
												jointPositions=q_des,
												jointVels=dq_des)








