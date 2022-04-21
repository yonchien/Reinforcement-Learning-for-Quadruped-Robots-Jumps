from sys import path
try:
	path.append(r"/home/guillaume/Documents/casadi-linux-py36-v3.5.1-64bit")
	from casadi import *
except:
	raise ImportError("Install casadi, either download or try: pip install casadi")

import numpy as np
import numpy.matlib
from usc_learning.envs.car.system_matrices_bike import *
from usc_learning.envs.car.system_matrices_car import *
import sys
from datetime import datetime


USE_CAR = 0

class CarOpt(object):
	""" Class to solve NLP for kinematic car model motion """
	def __init__(self):
		self.defaultN = 10
		self.N = 10
		self.DT = 0.05
		self.T = self.DT * self.N
		# 8 states, 2 lagrange multiplers (one no skid constraint for front, and one for back wheel)
		self.NUM_STATES = 8 + 2 
		self.NUM_INPUTS = 2

		self.F = self.getFunction()
		self.w = []
		self.w0 = []
		self.lbw = []
		self.ubw = []
		self.J = 0
		self.gc = []
		self.lbg = []
		self.ubg = []
		self.last_sol = None
		self.last_torques = None
		self.printFlag = False

		self.w0_init = np.zeros(self.NUM_STATES)
		self.formulate_nlp()
		self.goal = [10,0,0]

	def getFunction(self):
		N = self.N
		T = self.T
		x = SX.sym('x',self.NUM_STATES)
		U = SX.sym('U',self.NUM_INPUTS)

		xb, yb, thb, thf =  x[0],  x[1],  x[2], x[3]
		dxb, dyb, dthb, dthf = x[4], x[5], x[6], x[7]

		u_steering, u_wheels =  U[0],  U[1]

		# get_params, might want to have these in own function eventually?
		Jb=1; J1=1; J1w = 1; mb = 1; m1 = 1;
		Jb = 0.005; J1=1e-6
		b_length = 1; b_width = 0.2;
		b_length = 0.325 # from inspecting URDF, b_width is 0.2 (0.1*2)

		mb = 4; m1 = 0.34055
		Jb = 0.05865; J1=0.0003; b_length = 0.325;

		J2=J1; J3=J1; J4=J1; J2w=J1w; J3w=J1w; J4w=J1w; m2=m1; m3=m1; m4=m1; g=9.8;
		wR=.1; fcoeff = 0.0; mu_k = 1.0;

		# model equations
		if USE_CAR:
			print('using car equations')
			D = D_wlag_simple_rc(J1,J4,Jb,b_length,b_width,m1,m2,m3,m4,mb,thb,thf)
			C = C_wlag_simple_rc(b_length,b_width,dthb,dthf,dxb,dyb,m1,m2,m3,m4,thb,thf)
			Bu = Bu_w_lag_simple_rc(b_length,b_width,thb,thf,u_steering,u_wheels)
		else:
			D = D_wlag_simple_bike(J1,Jb,b_length,m1,m2,mb,thb,thf)
			C = C_wlag_simple_bike(b_length,dthb,dthf,dxb,dyb,m1,m2,thb,thf)
			Bu = Bu_w_lag_simple_bike(b_length,thb,thf,u_steering,u_wheels)

		# casadi format
		D = SX(D)
		C = SX(C)
		Bu = SX(Bu)
		# left side
		d2X = solve(D, (-C + Bu )) 
		dx = vertcat(x[4:8],d2X) 

		# Cost of Transport
		T_w = U 
		L_1 = T_w * T_w
		# sum1 to sum rows, sum2 for columns
		L = sum1(sqrt(L_1 + 0.000001))

		# Set up functions 
		M = 1
		DT = T/(N/M)
		f = Function('f', [x, U], [dx, L] )

		X0 = MX.sym('X0', self.NUM_STATES)
		X1 = MX.sym('X1', self.NUM_STATES)
		U0 = MX.sym('U0', self.NUM_INPUTS)

		X = X0
		Q = 0.0
		[k1, k1_q] = f(X1,U0)
		X = X + DT*k1
		Q = Q + DT*k1_q
		F = Function('F', [X0, X1, U0], [X,Q], ['x0', 'x1', 'p'], ['xf', 'qc'])

		return F


	def solveNLP(self):
		"""Solve NLP. """
		# Concatenate vectors
		w = vertcat(*self.w)
		gc = vertcat(*self.gc)
		w0 = np.concatenate(self.w0)
		lbw = np.concatenate(self.lbw)
		ubw = np.concatenate(self.ubw)
		lbg = np.concatenate(self.lbg)
		ubg = np.concatenate(self.ubg)

		print_shapes = False
		if print_shapes:
			print("w", w.shape)
			print("gc", gc.shape)
			print("w0", w0.shape)
			print("lbw", lbw.shape)
			print("ubw", ubw.shape)
			print("lbg", lbg.shape)
			print("ubg", ubg.shape)

		J = self.J
		N = self.N

		# Create an NLP solver
		prob = {'f': J, 'x': w, 'g': gc}
		if self.printFlag:
			optsipopt = {'fixed_variable_treatment': 'make_constraint', 'max_iter': 500000} 
			opts = {'ipopt': optsipopt, 'eval_errors_fatal': True}
		else:
			optsipopt = {'fixed_variable_treatment': 'make_constraint', 'max_iter': 500000, 'print_level': 0}
			#optsipopt['max_iter'] =  2
			opts = {'ipopt': optsipopt, 'eval_errors_fatal': True, 'print_time': 0}

		solver = nlpsol('solver', 'ipopt',prob, opts)
		# Solve the NLP
		sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
		# format solution
		w_opt = sol['x'].full()
		j_opt = sol['f'].full()

		x_init = w_opt[0:self.NUM_STATES]
		w_opt = w_opt[self.NUM_STATES:]

		u_opt = w_opt[:self.NUM_INPUTS*N]
		x_opt = w_opt[self.NUM_INPUTS*N:]

		self.last_sol = x_opt.flatten().copy()
		self.last_torques = u_opt.flatten().copy()

		# reshape x_opt
		x_opt = x_opt.reshape((self.NUM_STATES, N), order="F")
		x_opt = np.hstack( [ x_init.reshape((-1,1)), x_opt ])
		u_opt = u_opt.reshape((self.NUM_INPUTS, N), order="F")
		
		end_state = x_opt[:,-1].flatten()
		# Update trajectory length depending on how close we predict we will be to the goal.
		if np.linalg.norm( np.array(self.goal[0:2]) - end_state[0:2] ) < .5:
			self.updateN(max(round(self.N/2),1))
			#print('updated low', self.N)
		else:
			if self.N != self.defaultN:
				self.updateN(min(self.N*2,self.defaultN))
				#print('updated high', self.N)

		return x_opt, u_opt


	def updateN(self, N):
		"""Update trajectory length (number of knot points). 
		Must also reformulate NLP and set new goal location to be last knot point.
		"""
		self.N = N
		self.T = N * self.DT
		self.formulate_nlp()
		self.setJ(xgoal=self.goal[0], ygoal=self.goal[1], thgoal=self.goal[2])

	def formulate_nlp(self):
		""" set w, lbw, ubw, gc, lbg, ubg """
		w = []
		w0 = []
		lbw = []
		ubw = []
		J = 0
		gc = []
		lbg = []
		ubg = []
		N = self.N
		w0_init = self.w0_init

		Xi = MX.sym('Xi', self.NUM_STATES)
		w.append(Xi)
		w0.append(w0_init)
		lbw.append(w0_init)
		ubw.append(w0_init)

		# input bounds
		Uk = MX.sym('Uk',N*self.NUM_INPUTS)
		w.append(Uk)
		w0.append(np.zeros(N*self.NUM_INPUTS))
		torque_wheels = 2 #can also try 0.5, 1, 5
		torque_steering = 5
		lbw.append( numpy.matlib.repmat( - np.array([torque_steering, torque_wheels]), 1, N )[0] )
		ubw.append( numpy.matlib.repmat(   np.array([torque_steering, torque_wheels]), 1, N )[0] )

		self.sumsqr_Uks = sumsqr(Uk)

		# Formulate the NLP
		Xk = Xi

		for k in range(N):

			Xk1 = MX.sym('X_' + str(k+1), self.NUM_STATES)
			w.append(Xk1)

			pos_limits_low = np.array([-20,-20.0,-2*np.pi, -0.7])
			pos_limits_upp = np.array([ 20, 20.0, 2*np.pi,  0.7])
			#vel_limits_low = np.array( [-4, -4, -4, -4] )
			#vel_limits_upp = np.array( [ 4,  4,  4,  4] )
			vel_limits_low = np.array( [-3, -3, -2, -2] )
			vel_limits_upp = np.array( [ 3,  3,  2,  2] )

			lbw.extend([pos_limits_low, vel_limits_low, -1000*np.ones(2) ]) # added lagrange bounds, def won't be this high
			ubw.extend([pos_limits_upp, vel_limits_upp,  1000*np.ones(2) ])

			# initial conditions are same as original, might want to warm-start with existing trajectories afterwards
			w0.append(w0_init)
			# instead of w0_init, can random sample within the range (note: using previous solved trajectory after 1st run)
			#posk_inits = np.random.uniform(pos_limits_low,pos_limits_upp)
			#velk_inits = np.random.uniform(vel_limits_low,vel_limits_upp)
			#w0.append(np.hstack([posk_inits,velk_inits]))

			# Integrate until the end of the interval
			Fk = self.F(x0=Xk, x1=Xk1, p=Uk[self.NUM_INPUTS*k:self.NUM_INPUTS*k+self.NUM_INPUTS])
			Xk_end = Fk['xf']
			J = J + Fk['qc']

			# Equality constraint for integration
			# Need beginning of next state to match end of previous state
			gc.append(Xk_end - Xk1)
			lbg.append(np.zeros(self.NUM_STATES))
			ubg.append(np.zeros(self.NUM_STATES))

			Xk = Xk1

		self.lastXk = Xk # so we can use this later for goal without worrying about indexing
		self.w = w
		self.w0 = w0
		self.lbw = lbw
		self.ubw = ubw
		self.J = J
		self.gc = gc
		self.lbg = lbg
		self.ubg = ubg


	def set_w0(self, curr_state):
		"""Set initial conditions for optimization (warm start). """
		# use current state as initial condition
		# default is w0_init, update w/ curr_state
		if curr_state is None:
			curr_state = self.w0_init
		N = self.N
		DT = self.DT
		self.w0_init = curr_state
		w0_init = curr_state
		w0 = []
		#w0[0] = curr_state
		self.lbw[0] = curr_state
		self.ubw[0] = curr_state
		w0.append(curr_state) # init conditions

		INIT_FLAG = True
		if INIT_FLAG and self.last_sol is not None: 
			# check if length of last_sol == N*self.NUM_INPUTS
			#	-if so, set to same
			index_diff = int(N*self.NUM_INPUTS - self.last_torques.shape[0])
			# do the same for the states	
			index_diff_states = int(N*self.NUM_STATES - self.last_sol.shape[0])

			if index_diff == 0:
				#print('same len')
				w0.append(self.last_torques)
				w0.append(self.last_sol)
			elif index_diff > 0:
				#print('len(last_sol) < N*NUM_INPUTS')
				w0.append(self.last_torques)
				w0.append(numpy.matlib.repmat( self.last_torques[-self.NUM_INPUTS:], 1, int(index_diff / self.NUM_INPUTS) )[0] )
				w0.append(self.last_sol)
				w0.append(numpy.matlib.repmat( self.last_sol[-self.NUM_STATES:], 1, int(index_diff_states / self.NUM_STATES) )[0] )
			# if len(last_sol > N*self.NUM_INPUTS
			# use chunk of last_sol
			elif index_diff < 0:
				#print('len(last_sol) > N*NUM_INPUTS')
				w0.append(self.last_torques[:N*self.NUM_INPUTS])
				w0.append(self.last_sol[:N*self.NUM_STATES])

		else: 
			# zeros for all N (inputs)
			w0.append(np.zeros(N*self.NUM_INPUTS))
			# repeat initial states
			w0.append(numpy.matlib.repmat(w0_init,1,N )[0])

		self.w0 = w0

	def setJ(self,xgoal=None, ygoal=None, thgoal=None):
		"""Set cost function, minimizes distance of last knot point to [x,y,thb]"""
		self.goal = [xgoal,ygoal,thgoal]
		if xgoal is None:
			xgoal = self.lastXk[0]
		if ygoal is None:
			ygoal = self.lastXk[1]
		if thgoal is None:
			thgoal = self.lastXk[2]

		self.J = 10*(self.lastXk[0] - xgoal)**2 + 10*(self.lastXk[1] - ygoal)**2 + 1000*(self.lastXk[2] - thgoal)**2 #+ 1e-2*self.sumsqr_Uks


	def testF(self):
		"""Test function correctly setup (or at least no error). """
		Fk = self.F(x0=np.arange(self.NUM_STATES)+1, x1=np.arange(self.NUM_STATES)+1, p=np.arange(self.NUM_INPUTS)+1)
		print("Fk['xf']",Fk['xf'])
		print("Fk['qc']", Fk['qc'])



if __name__ == '__main__':
	optObj = CarOpt()
	optObj.testF()
	optObj.setJ(xgoal=0.001, ygoal=3, thgoal=0)
	# optObj.printFlag = True
	startTime = datetime.now()
	for j in range(100):
		temptime = datetime.now() 
		x_opt, u_opt = optObj.solveNLP()
		print(datetime.now() - temptime)
		#print(x_opt)
		#print(u_opt)
		#print(x_opt[:,1].flatten())
		optObj.set_w0(x_opt[:,1].flatten())
		#optObj.set_w0(optObj.w0_init)
	print(datetime.now() - startTime)
	#optObj.solveNLP()

