import sys

from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
from stable_baselines import PPO2,A2C,DDPG

model = PPO2.load("logs/ppo2/052020033438/model_ckpts/aliengo_ppo2_23600000_steps")

# Enjoy trained agent
env = AlienGoGymEnv(isrender = True)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
    	obs = env.reset()
#     env.render()