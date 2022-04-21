import os
import copy
import math
import numpy as np

"""Some modifications to normal pybullet racecar.py """

class RcCar:

	def __init__(self, bullet_client, urdfRootPath='', timeStep=0.01):
		self.urdfRootPath = urdfRootPath
		self.timeStep = timeStep
		self._p = bullet_client
		self.reset()

	def reset(self):
		#car = self._p.loadURDF(os.path.join(self.urdfRootPath,"racecar/racecar_differential.urdf"), [0,0,.2],useFixedBase=False)

		car = self._p.loadURDF(os.path.join(self.urdfRootPath,"racecar/racecar_differential_me.urdf"), [-0.1625,0,.05],useFixedBase=False)

		self.racecarUniqueId = car
		#for i in range (self._p.getNumJoints(car)):
		#	print (self._p.getJointInfo(car,i))
		for wheel in range(self._p.getNumJoints(car)):
				self._p.setJointMotorControl2(car,wheel,self._p.VELOCITY_CONTROL,targetVelocity=0,force=0)
				self._p.getJointInfo(car,wheel)

		#self._p.setJointMotorControl2(car,10,self._p.VELOCITY_CONTROL,targetVelocity=1,force=10)
		c = self._p.createConstraint(car,9,car,11,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=1, maxForce=10000)

		c = self._p.createConstraint(car,10,car,13,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

		c = self._p.createConstraint(car,9,car,13,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

		c = self._p.createConstraint(car,16,car,18,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=1, maxForce=10000)

		c = self._p.createConstraint(car,16,car,19,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

		c = self._p.createConstraint(car,17,car,19,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

		c = self._p.createConstraint(car,1,car,18,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
		c = self._p.createConstraint(car,3,car,19,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
		self._p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15,maxForce=10000)

		self.steeringLinks = [0,2]
		self.maxForce = 100 # orig 100
		self.nMotors = 2
		self.motorizedwheels=[8,15]
		self.speedMultiplier = 80.
		self.steeringMultiplier = 0.7

	def getActionDimension(self):
		return self.nMotors

	def get_w0_state(self):
		# this is for the optimization, initial state w0
		# [xb, yb, thb, thf, dxb, dyb, dthb, dthf, lambda1, lambda2]
		# dummy link is at index 20
		# these latter 4 are world frame link pos/orn/linVel/angVel
		_, _, _, _, pos, orn, linVel, angVel = self._p.getLinkState(self.racecarUniqueId, 20, 
								computeLinkVelocity=True, computeForwardKinematics=True)
		roll, pitch, yaw = self._p.getEulerFromQuaternion(orn)

		steering_jointStates =  self._p.getJointStates(self.racecarUniqueId, self.steeringLinks)
		steering_jointPositions = np.array([x[0] for x in steering_jointStates])
		steering_jointVelocities = np.array([x[1] for x in steering_jointStates]) 

		w0 = np.array([pos[0],pos[1],yaw, np.mean(steering_jointPositions), linVel[0], linVel[1], angVel[2], np.mean(steering_jointVelocities), 0.,0. ])
		return w0


	def applyAction(self, motorCommands):
		"""Set velocity for differential ring, and steering angle. """
		targetVelocity= motorCommands[0]*self.speedMultiplier
		steeringAngle = motorCommands[1]*self.steeringMultiplier

		for motor in self.motorizedwheels:
			self._p.setJointMotorControl2(self.racecarUniqueId,motor,self._p.VELOCITY_CONTROL,targetVelocity=targetVelocity,force=self.maxForce)
		for steer in self.steeringLinks:
			self._p.setJointMotorControl2(self.racecarUniqueId,steer,self._p.POSITION_CONTROL,targetPosition=steeringAngle,force=self.maxForce)
