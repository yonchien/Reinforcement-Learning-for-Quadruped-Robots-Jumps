from usc_learning.envs.aliengo import AlienGo 
from usc_learning.envs.aliengo import TORQUE_LIMITS, HIP_POSITIONS
import numpy as np
import pybullet as pyb
import time

""" Test impedance control standing - messy original. See test_impedance_control.py """

robot = AlienGo(pybullet_client=pyb,render=True,control_mode="TORQUE")
time.sleep(0.5)
robot.SetPoseFromHeight(0.4)
time.sleep(0.5)
pyb.stepSimulation()
print(robot.GetContactInfo())

hip_offset = 0.083
#HIP_POSITIONS # 0.051
FL_thigh_from_hip = np.array([0,hip_offset,0])
THIGH_POSITIONS = [] #0.083

# FR_hip_reset = np.array([ 0.2399 , -0.11, 0])
# FL_hip_reset = np.array([ 0.2399 ,  0.11, 0])
# RR_hip_reset = np.array([-0.2399 , -0.11, 0])
# RL_hip_reset = np.array([-0.2399 ,  0.11, 0])

#HIP_POSITIONS = [FR_hip_reset, FL_hip_reset, RR_hip_reset, RL_hip_reset]
# THIGH_POSITIONS[0] = HIP_POSITIONS[0] - FL_thigh_from_hip 
# THIGH_POSITIONS[1] = HIP_POSITIONS[1] + FL_thigh_from_hip
# THIGH_POSITIONS[2] = HIP_POSITIONS[2] - FL_thigh_from_hip
# THIGH_POSITIONS[3] = HIP_POSITIONS[3] + FL_thigh_from_hip 

#HIP_POSITIONS = HIP_POSITIONS_RESET
print(HIP_POSITIONS)


kpCartesian = 1000*np.eye(3)
kdCartesian = 20*np.eye(3) 
kpCartesian = np.eye(3) @ np.array([500,500,50])
kdCartesian = np.eye(3) @ np.array([20,20,20])

zd_hip = 0.4
Pcd = np.array([0,0,0.4])
Rd = np.eye(3)

Pd = np.array([0,-hip_offset,-zd_hip])
Pds = robot.FormatFRtoAll(Pd)
print("Pds",Pds)

vd = np.zeros(3)
Fdw = np.array([ 0, 0, -robot.total_mass*(9.8/4) ])
#Fd = robot.GetBaseOrientationMatrix() @ 

#Pd_hip = 

# p_foot_baseFrame = robot.GetLocalEEPos(i)
# p_hip = HIP_POSITIONS[i]
# p_foot_hipFrame = p_foot_baseFrame - p_hip

while True:
	print('time {:.3f}'.format(robot.GetSimTime()))
	action = np.zeros(12)
	# loop through hips
	for i in range(4):
		# body position and orientation matrix
		basePos = robot.GetBasePosition()
		R = robot.GetBaseOrientationMatrix() 
		baseLinvel = np.asarray(robot.GetBaseLinearVelocity())
		

		p_foot_baseFrame = robot.GetLocalEEPos(i)
		p_hip = HIP_POSITIONS[i]# basePos + R @ HIP_POSITIONS[i]
		p_foot_hipFrame = p_foot_baseFrame - p_hip

		# if i==0:
		# 	print('hip', p_hip)
		# 	print('foot',p_foot_baseFrame)
		# 	print('foot hip', p_foot_hipFrame)
		# 	print("\n")

		# Jacobian and current foot position
		J, ee_pos_legFrame = robot._ComputeJacobianAndPositionUSC(i)
		linJac, angJac = robot._CalculateJacobian(i)

		#print(p_foot_hipFrame, ee_pos_legFrame)
		#assert np.isclose(p_foot_hipFrame, ee_pos_legFrame).all() # not always true
		
		Pd_hipw = Pcd + Rd @ HIP_POSITIONS[i]
		#Pd = R.T @ (p_foot_hipFrame - Pd_hipw)
		Pd =   Pds[i] #np.array([0,0,-zd_hip])
		Fd = R.T @ Fdw

		#print(p_foot_hipFrame, ee_pos_legFrame) # NOT THE SAME!!

		foot_linvel = robot.GetLocalEEVel(i)
		foot_linvel,_ = robot.GetGlobalEEVel(i)
		if i==0:
			print("\n")
			print('base vel', baseLinvel)
			print('local foot linvel',foot_linvel)
			print('Jacobian ',J @ robot.GetJointVelocities()[i*3:i*3+3])
			#print('relative linvel', R.T @ (foot_linvel - baseLinvel))

		#foot_linvel = J @ robot.GetJointVelocities()[i*3:i*3+3]
		#print("Pd", Pd)
		#print('foot pos', ee_pos_legFrame)
		tau = J.T @ ( Fd + kpCartesian @ (Pd-ee_pos_legFrame) + kdCartesian @ (vd-foot_linvel))
		action[i*3:i*3+3] = tau

	#action=(2*np.random.random(12)-1) * TORQUE_LIMITS
	robot.Step(action)
	#print(robot.GetJointAngles())
	#print(robot.GetBasePosition() )

	time.sleep(0.05)