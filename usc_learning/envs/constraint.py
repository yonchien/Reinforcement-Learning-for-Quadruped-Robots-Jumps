# import pybullet as p
# import time
# import math

# import pybullet_data
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# p.loadURDF("plane.urdf")
# cubeId = p.loadURDF("cube_small.urdf", 0, 0, 1)
# p.setGravity(0, 0, -10)
# p.setRealTimeSimulation(1)
# cid = p.createConstraint(cubeId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
# print(cid)
# print(p.getConstraintUniqueId(0))
# a = -math.pi
# while 1:
#   a = a + 0.01
#   if (a > math.pi):
#     a = -math.pi
#   time.sleep(.01)
#   p.setGravity(0, 0, -10)
#   pivot = [a, 0, 1]
#   orn = p.getQuaternionFromEuler([a, 0, 0])
#   p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)

# p.removeConstraint(cid)
import time
import numpy as np
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv

env = QuadrupedGymEnv(render=True)
p = env._pybullet_client

quad_base = np.array(env._robot.GetBasePosition())
quad_ID = env._robot.quadruped

block_pos_delta_base_frame = -1*np.array([-0.2, 0.1, -0.])

sh_colBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05]*3)
orn = p.getQuaternionFromEuler([0,0,0])
base_block_ID=p.createMultiBody(baseMass=5,
								baseCollisionShapeIndex = sh_colBox,
                    			basePosition = quad_base + block_pos_delta_base_frame,#[quad_base[0], quad_base[1], quad_base[2]+0.3],
                    			baseOrientation=[0,0,0,1])#orn)

cid = p.createConstraint(quad_ID, -1, base_block_ID, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)
# disable self collision between box and each link
for i in range(-1,p.getNumJoints(quad_ID)):
	p.setCollisionFilterPair(quad_ID,base_block_ID, i,-1, 0)
#[0.1, 0, -0.1])
time.sleep(0.2)
while 1:
	a = np.random.random(6)
	a = np.zeros(6)
	env.step(a)