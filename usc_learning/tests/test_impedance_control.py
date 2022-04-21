import numpy as np
import pybullet as pyb
import time

from usc_learning.envs.aliengo import AlienGo 
from usc_learning.envs.aliengo import TORQUE_LIMITS, HIP_POSITIONS
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

""" Test impedance control standing. """

robot = AlienGo(pybullet_client=pyb,render=False,control_mode="TORQUE",time_step=0.001,actionRepeat=1)
robot.SetPoseFromHeight(0.4)
time.sleep(0.5)
pyb.stepSimulation()

# kp and kd gains
kpCartesian = 1000*np.eye(3)
kdCartesian = 20*np.eye(3) 
# kpCartesian = np.eye(3) @ np.array([500,500,200])
# kdCartesian = np.eye(3) @ np.array([50,50,20])
kpCartesian = np.diag([800,2500,2000])
kdCartesian = np.diag([30,30,30])

kpCartesian = np.diag([800,1400,1400])

# set desired foot position for FR, and map correspondingly to other feet
hip_offset = 0.083
zd_hip = 0.4
Pd = np.array([0,-hip_offset,-zd_hip])
Pds = robot.FormatFRtoAll(Pd)

# desired foot velocity 
vd = np.zeros(3)
# desired force
Fdw = np.array([ 0, 0, -robot.total_mass*(9.8/4) ])
#Fdw = np.array([ 0, 0, -46.5 ])
print(robot.total_mass*(9.8/4)) # 50.5631245
#sys.exit()

while True:
	action = np.zeros(12)
	# body position, orientation matrix, velocity
	basePos = np.asarray(robot.GetBasePosition())
	R = robot.GetBaseOrientationMatrix() 
	baseLinvel = np.asarray(robot.GetBaseLinearVelocity())
	baseOrn = np.asarray(robot.GetBaseOrientationRollPitchYaw())
	print('time {:.3f}'.format(robot.GetSimTime()))
	print('base pos', basePos)
	print('base vel', baseLinvel)
	print('base rpy', baseOrn)

	# loop through hips
	for i in range(4):
		# Jacobian and current foot position in hip frame
		J, ee_pos_legFrame = robot._ComputeJacobianAndPositionUSC(i)
		# desired foot position
		Pd = Pds[i] 
		# desired force (rotated)
		Fd = R.T @ Fdw

		# foot velocity in leg frame
		foot_linvel = J @ robot.GetJointVelocities()[i*3:i*3+3]
		# global foot velocity
		#foot_linvel,_ = robot.GetGlobalEEVel(i)

		tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))

		#tau = J.T @ ( kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))

		action[i*3:i*3+3] = tau

	# step simulation
	robot.Step(action)
