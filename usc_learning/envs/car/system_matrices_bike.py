import numpy as np
from sys import path
path.append(r"/home/guillaume/Documents/casadi-linux-py36-v3.5.1-64bit")
from casadi import *


def D_wlag_simple_bike(J1,Jb,b_length,m1,m2,mb,thb,thf):
# %D_WLAG_SIMPLE_BIKE
# %    DNEW2 = D_WLAG_SIMPLE_BIKE(J1,JB,B_LENGTH,M1,M2,MB,THB,THF)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    23-Apr-2019 10:29:04

	D = np.array([m1+m2+mb,0.0,b_length*m1*sin(thb)*(-1.0/2.0)+b_length*m2*sin(thb)*(1.0/2.0),0.0,-sin(thb+thf),-sin(thb),0.0,m1+m2+mb,b_length*m1*cos(thb)*(1.0/2.0)-b_length*m2*cos(thb)*(1.0/2.0),0.0,cos(thb+thf),cos(thb),b_length*m1*sin(thb)*(-1.0/2.0)+b_length*m2*sin(thb)*(1.0/2.0),b_length*m1*cos(thb)*(1.0/2.0)-b_length*m2*cos(thb)*(1.0/2.0),J1+Jb+b_length**2*m1*(1.0/4.0)+b_length**2*m2*(1.0/4.0),J1,b_length*cos(thf)*(1.0/2.0),b_length*(-1.0/2.0),0.0,0.0,J1,J1,0.0,0.0,sin(thb+thf),-cos(thb+thf),b_length*cos(thf)*(-1.0/2.0),0.0,0.0,0.0,sin(thb),-cos(thb),b_length*(1.0/2.0),0.0,0.0,0.0]).reshape((6,6),order="F")

	return D

def C_wlag_simple_bike(b_length,dthb,dthf,dxb,dyb,m1,m2,thb,thf):
# %C_WLAG_SIMPLE_BIKE
# %    CNEW2 = C_WLAG_SIMPLE_BIKE(B_LENGTH,DTHB,DTHF,DXB,DYB,M1,M2,THB,THF)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    23-Apr-2019 10:29:04

	C = np.array([b_length*dthb**2*cos(thb)*(m1-m2)*(-1.0/2.0),b_length*dthb**2*sin(thb)*(m1-m2)*(-1.0/2.0),0.0,0.0,-dthb*dxb*cos(thb+thf)-dthf*dxb*cos(thb+thf)-dthb*dyb*sin(thb+thf)-dthf*dyb*sin(thb+thf)-b_length*dthb*dthf*sin(thf)*(1.0/2.0),-dthb*(dxb*cos(thb)+dyb*sin(thb))]).reshape((6,1))

	return C

def Bu_w_lag_simple_bike(b_length,thb,thf,u_steering,u_wheels):
# %BU_W_LAG_SIMPLE_BIKE
# %    BUNEW2 = BU_W_LAG_SIMPLE_BIKE(B_LENGTH,THB,THF,U_STEERING,U_WHEELS)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    23-Apr-2019 10:29:04

	Bu = np.array([u_wheels*(cos(thb)*cos(thf)-sin(thb)*sin(thf))+u_wheels*cos(thb),u_wheels*(cos(thb)*sin(thf)+cos(thf)*sin(thb))+u_wheels*sin(thb),b_length*u_wheels*cos(thb)*(cos(thb)*sin(thf)+cos(thf)*sin(thb))*(1.0/2.0)-b_length*u_wheels*sin(thb)*(cos(thb)*cos(thf)-sin(thb)*sin(thf))*(1.0/2.0),u_steering,0.0,0.0]).reshape((6,1))

	return Bu


def D_bike_fric(J1,Jb,b_length,m1,m2,mb,thb):
# %D_BIKE_FRIC
# %    D = D_BIKE_FRIC(J1,JB,B_LENGTH,M1,M2,MB,THB)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    20-Aug-2019 14:28:51
	D = np.array([m1+m2+mb,0.0,b_length*m1*sin(thb)*(-1.0/2.0)+b_length*m2*sin(thb)*(1.0/2.0),0.0,0.0,m1+m2+mb,b_length*m1*cos(thb)*(1.0/2.0)-b_length*m2*cos(thb)*(1.0/2.0),0.0,b_length*m1*sin(thb)*(-1.0/2.0)+b_length*m2*sin(thb)*(1.0/2.0),b_length*m1*cos(thb)*(1.0/2.0)-b_length*m2*cos(thb)*(1.0/2.0),J1+Jb+b_length**2*m1*(1.0/4.0)+b_length**2*m2*(1.0/4.0),J1,0.0,0.0,J1,J1]).reshape((4,4),order="F")

	return D


def C_bike_fric(b_length,dthb,m1,m2,thb):
# %C_BIKE_FRIC
# %    C = C_BIKE_FRIC(B_LENGTH,DTHB,M1,M2,THB)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    20-Aug-2019 14:28:51

	C = np.array([b_length*dthb**2*cos(thb)*(m1-m2)*(-1.0/2.0),b_length*dthb**2*sin(thb)*(m1-m2)*(-1.0/2.0),0.0,0.0]).reshape((4,1))
	return C

def Bu_bike_fric(b_length,thb,thf,u_steering,u_wheels):
# %BU_BIKE_FRIC
# %    BU = BU_BIKE_FRIC(B_LENGTH,THB,THF,U_STEERING,U_WHEELS)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    20-Aug-2019 14:28:51

	Bu = np.array([u_wheels*(cos(thb+thf)+cos(thb)),u_wheels*(sin(thb+thf)+sin(thb)),b_length*u_wheels*sin(thf)*(1.0/2.0),u_steering]).reshape((4,1))
	return Bu

def F_bike_fric(Ffric_x1,Ffric_x2,Ffric_y1,Ffric_y2,b_length,thb):
# %F_BIKE_FRIC
# %    F = F_BIKE_FRIC(FFRIC_X1,FFRIC_X2,FFRIC_Y1,FFRIC_Y2,B_LENGTH,THB)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    20-Aug-2019 14:28:51

	F = np.array([Ffric_x1+Ffric_x2,Ffric_y1+Ffric_y2,Ffric_y1*b_length*cos(thb)*(1.0/2.0)-Ffric_y2*b_length*cos(thb)*(1.0/2.0)-Ffric_x1*b_length*sin(thb)*(1.0/2.0)+Ffric_x2*b_length*sin(thb)*(1.0/2.0),0.0]).reshape((4,1))
	return F

def front_wheel_frame_mapping_bike_fric(b_length,thb,thf):
# %FRONT_WHEEL_FRAME_MAPPING_BIKE_FRIC
# %    FRONT_WHEEL_FRAME_MAPPING = FRONT_WHEEL_FRAME_MAPPING_BIKE_FRIC(B_LENGTH,THB,THF)

# %    This function was generated by the Symbolic Math Toolbox version 8.1.
# %    20-Aug-2019 14:28:52

	front_wheel_frame_mapping = np.array([cos(thb+thf),sin(thb+thf),b_length*sin(thf)*(1.0/2.0),0.0,-sin(thb+thf),cos(thb+thf),b_length*cos(thf)*(1.0/2.0),0.0]).reshape((4,2),order="F")
	return front_wheel_frame_mapping



