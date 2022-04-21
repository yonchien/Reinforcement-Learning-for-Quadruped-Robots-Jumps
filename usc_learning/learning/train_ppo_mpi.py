"""
trains quadruped gym env with MPI (fast wallclock time)
run with
$ mpirun -np 96 python train_ppo_mpi.py (replace 96 with number of CPU cores you have.)
should also update timesteps_per_actorbatch accordingly, about 4096 per upate is good
  so, for 4 cores - 1024, 8 cores - 512, etc.
$ mpirun -np 4 python train_ppo_mpi.py

"""

import os
import gym
#import slimevolleygym
from datetime import datetime

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import bench, logger, PPO1, PPO2
from stable_baselines.common.callbacks import EvalCallback
import tensorflow as tf

# Vec Normalize related
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.cmd_util import make_vec_env

from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
from usc_learning.utils.learning_utils import CheckpointCallback
from usc_learning.learning.algos.ppo2_mpi import PPO2_MPI

NUM_TIMESTEPS = int(1e7)
SEED = 831
EVAL_FREQ = 200000
EVAL_EPISODES = 1000
LOGDIR = "ppo1_mpi"

save_path = 'logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(save_path, exist_ok=True)
LOGDIR = save_path

def make_env(seed,_log_path):
  # env = gym.make("SlimeVolley-v0")
  #env = gym.make("QuadrupedGymEnv-v0")
  #env = Monitor(QuadrupedGymEnv(render=False), filename=save_path+'monitor.csv', allow_early_resets=True)

  #env = QuadrupedGymEnv(render=False)
  #env = DummyVecEnv( [lambda: env])

  env = lambda: QuadrupedGymEnv(render=False)
  env = make_vec_env(env, monitor_dir=_log_path,n_envs=1) #
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)
  env.seed(seed)
  return env

def train():
  """
  Train PPO1 model for slime volleyball, in MPI multiprocessing. Tested for 96 CPUs.
  """
  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
    logger.configure(folder=LOGDIR)

  else:
    logger.configure(format_strs=[])
  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  #env = make_env(workerseed,rank)
  env = make_env(workerseed,logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

  #env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
  env.seed(workerseed)

  # network architecture
  policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200,100])

  # model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=1024, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
  #              optim_stepsize=1e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='constant',policy_kwargs=policy_kwargs,
  #              verbose=1)

  """
  n_steps = 1024
  nminibatches = int(n_steps / 64)
  learning_rate = lambda f: 1e-4 #5e-4 * f
  ppo_config = {"gamma":0.99, "n_steps":n_steps, "ent_coef":0.0, "learning_rate":learning_rate, "vf_coef":0.5,
          "max_grad_norm":0.5, "lam":0.95, "nminibatches":nminibatches, "noptepochs":10, "cliprange":0.2, "cliprange_vf":None,
          "verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":policy_kwargs,
          "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None
  """
  # just learning 3 separate things currently, since no MPI statements here
  # TODO: write PPO2_mpi
  model = PPO2_MPI(MlpPolicy, env, n_steps=1024, cliprange=0.2, ent_coef=0.0, noptepochs=10,
             learning_rate=1e-4, nminibatches=16, gamma=0.99, lam=0.95, policy_kwargs=policy_kwargs,
             vf_coef=0.5, cliprange_vf=None, max_grad_norm=0.5,
             verbose=1)

  #eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)
  eval_callback = CheckpointCallback(save_freq=10000, save_path=save_path,name_prefix='rl_model', verbose=2)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  env.close()
  del env
  if rank == 0:
    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.


if __name__ == '__main__':
  train()