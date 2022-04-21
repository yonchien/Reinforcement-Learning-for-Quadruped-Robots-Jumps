import numpy as np
from carOpt import CarOpt
from usc_learning.envs.car.rcCarFlagRunGymEnv import RcCarFlagRunGymEnv, angle_between, rot_mat


################################################################################################
# Helper class for calling MPC
################################################################################################
class SolveMPC(object):
	""" Helper class for calling CarOpt """
	def __init__(self):
		self.optObj = CarOpt()
		self.goal = None

	def findTrajOptAction(self,w0):
		"""Take in init state for MPC, set initial condition, and solve MPC, 
		returning actions to be supplied to environment. """
		#w0 = self._racecar.get_w0_state()
		self.optObj.set_w0(w0)
		x_opt, u_opt = self.optObj.solveNLP()
		next_state = x_opt[:,1].flatten()

		return self.format_action(w0,next_state)

	def updateGoal(self,goal):
		self.goal = goal
		self.optObj.setJ(xgoal=goal[0],ygoal=goal[1])

	def getGoal(self):
		return self.goal

	def format_action(self,w0,next_state):
		""" Based on current state and next desired state, convert optimal action to [-1,1] range. """
		xb, yb, thb, thf = next_state[0], next_state[1], next_state[2], next_state[3]
		dxb, dyb, dthb, dthf = next_state[4], next_state[5], next_state[6], next_state[7]

		body_vel_magnitude = np.linalg.norm(next_state[4:6])
		body_vel_direction = np.arctan2(next_state[5] , next_state[4] )

		# fix for very small velocities in x or y directions, otherwise getting opposite of what we want
		my_eps = 1e-3
		if np.abs(dxb) < my_eps and np.abs(dyb) < my_eps:
			body_vel_direction = 0
		elif np.abs(dyb) < my_eps:
			body_vel_direction = np.arctan2(0, dxb)
		elif np.abs(dxb) < my_eps:
			body_vel_direction = np.arctan2(dyb,0)
		else:
			body_vel_direction = np.arctan2(dyb,dxb)

		# direction body and velocity of body are facing in planar vector form
		body_xy_vec = np.matmul( rot_mat(thb), np.array([[1],[0]]) )
		body_vel_vec = np.matmul( rot_mat(body_vel_direction), np.array([[1],[0]]) )
		body_xy_vec = body_xy_vec.reshape(2,)
		body_vel_vec = body_vel_vec.reshape(2,)

		ang_between = angle_between(body_xy_vec, body_vel_vec)

		if np.abs(ang_between) < np.pi/2:
			vel_dir = 1
		else:
			vel_dir = -1

		# experimentally determined factors, pybullet car has gear train and transmission 
		# to map from differential drive to body velocity
		targetVelocity = vel_dir * body_vel_magnitude * 21 #* 1.3
		steeringAngle = next_state[3] #+ ()

		targetVelocity = targetVelocity + 100*vel_dir * ( body_vel_magnitude - np.linalg.norm( w0[4:6] ) )
		steeringAngle = steeringAngle + ( next_state[3] - w0[3] )

		# divide by multipliers, since they will be scaled back up in applyAction
		targetVelocity = targetVelocity / 80. #self._racecar.speedMultiplier
		steeringAngle = steeringAngle / 0.7 #self._racecar.steeringMultiplier

		return np.array([targetVelocity, steeringAngle])

################################################################################################
# Test MPC
################################################################################################
# set up car env 
env = RcCarFlagRunGymEnv(render=True,record_video=False)
# get initial information from environment - goal and car state
goal = env.get_goal()
car_state = env.get_optimization_state()
# set up MPC solver
solver = SolveMPC()
# set cost for MPC and find action
solver.updateGoal(goal)
action = solver.findTrajOptAction(car_state)


while True:
	#print('*'*80)
	obs, rew, done, info = env.step(action)
	goal = info['goal']
	car_state = info['car_state']
	if goal != solver.getGoal():
		solver.updateGoal(goal)
	action = solver.findTrajOptAction(car_state)

	if done:
		env.reset()