#!/usr/bin/env python
"""Load rllib. Pretty hacky. """
import argparse
import collections
import copy
import json
import os, sys
from pathlib import Path
import pickle
import shelve
import time

import gym
import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env

#################################################################################################
#
#
# Please see load_rllib for a much more intuitive interface (not maintaining this file)
#
#
#################################################################################################

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# from ray.rllib.examples.env.parametric_actions_cartpole import \
#     ParametricActionsCartPole
# from ray.rllib.examples.model.parametric_actions_model import \
#     ParametricActionsModel
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))
#global log_dir, checkpoint, configs_file, alg
from usc_learning.utils.utils import plot_results, load_rllib
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
import numpy as np
# stable baselines vec env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from usc_learning.utils.file_utils import get_latest_model_rllib, get_latest_directory, read_env_config

# log_dir = './ray_results/PPO/' +\
#  'PPO_DummyQuadrupedEnv_0_2020-06-25_23-44-284ia1ha30/'
# checkpoint = log_dir + 'checkpoint_2300/checkpoint-2300'

#ray.shutdown()
log_dir = './ray_results/PPO/' + 'PPO_DummyQuadrupedEnv_0_2020-07-10_15-38-19coc11r2a/'#'PPO_DummyQuadrupedEnv_0_2020-07-08_19-42-15p9defkm9/'
checkpoint = get_latest_model_rllib(log_dir) #log_dir + 'checkpoint_340/checkpoint-340'
#data = load_rllib(log_dir)
alg = "PPO"

# log_dir = './ray_results/DDPPO/' + 'DDPPO_DummyQuadrupedEnv_0_2020-07-08_13-39-25d1ppl3p1/'
# log_dir = './ray_results/DDPPO/' + 'DDPPO_DummyQuadrupedEnv_0_2020-07-08_13-10-21dntmeec0/'
# checkpoint = log_dir + 'checkpoint_390/checkpoint-390'
# data = load_rllib(log_dir)
# alg= "DDPPO"

# log_dir = './ray_results/SAC/' + 'SAC_DummyQuadrupedEnv_0_2020-06-29_16-16-47kytack69/'
# checkpoint = log_dir + 'checkpoint_2250/checkpoint-2250'
# alg = 'SAC'
# plot_results([log_dir] , 10e10, 'timesteps', 'AlienGo PPO')
# plt.show() #block=False

#stats_path = os.path.join(log_dir, "vec_normalize.pkl")

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
stats_path = os.path.join(script_dir,get_latest_directory('ray_results/vec_monitor/'))
print(stats_path)
#sys.exit()

#stats_path = get_latest_directory('./ray_results/vec_monitor/')
#print(env_path)
#stats_path = os.path.join(env_path, "vec_normalize.pkl")
# get env config if available
try:
    # stats path has all saved files (quadruped, as well as vec)
    env_config = read_env_config(stats_path)
    sys.path.append(stats_path)
    print(sys.path)
    sys.exit()
    # check env_configs.txt file to see whether to load aliengo or a1
    with open(stats_path + '/env_configs.txt') as configs_txt:
        if 'aliengo' in configs_txt.read():
            import configs_aliengo as robot_config
        else:
            import configs_a1 as robot_config

    env_config['robot_config'] = robot_config
except:
    env_config = {}
env_config['render'] = True

print(env_config)
#sys.exit()
USING_VEC_ENV = True
#stats_path = './ray_results/vec_monitor/'
#stats_path = get_latest_directory(stats_path) + "/vec_normalize.pkl"
stats_path = os.path.join(stats_path,"vec_normalize.pkl")
print(stats_path)

class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, dummy_env_config):
        # set up like stable baselines (will this work?)
        #env = QuadrupedGymEnv(render=False,hard_reset=True)
        if USING_VEC_ENV:
            env = lambda: QuadrupedGymEnv(**env_config) #render=True,hard_reset=True
            env = make_vec_env(env, n_envs=1)
            env = VecNormalize.load(stats_path, env)
            # do not update stats at test time
            env.training = False
            # reward normalization is not needed at test time
            env.norm_reward = False

            self.env = env 
            self._save_vec_stats_counter = 0
        else:
            self.env = QuadrupedGymEnv(render=True,hard_reset=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print('\n','*'*80)
        print('obs high',self.observation_space.high)
        print('obs low', self.observation_space.low)

    def reset(self):
        """reset env """
        obs = self.env.reset()
        if USING_VEC_ENV:
            obs = obs[0]
            # no need to save anything at load time
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        """step env """
        if USING_VEC_ENV:
            obs, rew, done, info = self.env.step([action])
            obs = obs[0]
            rew = rew[0]
            done = done[0]
            info = info[0]
        else:
            obs, rew, done, info = self.env.step(action)
        #print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            sys.exit()
        return obs, rew, done, info 
#     # def render(self,):
#     #     self.env.render()

# needed for this file
register_env("quadruped_env",  lambda _: DummyQuadrupedEnv(_))


class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(
                    mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            pickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done])
        self._total_steps += 1


def create_parser(parser_creator=None):
    #global log_dir, checkpoint, configs_file, alg
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    # checkpoint_str = log_dir+checkpoint
    # parser.add_argument(
    #     "checkpoint", 
    #     type=str,
    #     default=checkpoint_str, 
    #     help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    # required_named.add_argument(
    #     "--run",
    #     type=str,
    #     default="PPO",#alg,
    #     required=True,
    #     help="The algorithm or model to train. This may refer to the name "
    #     "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
    #     "user-defined trainable function or class registered in the "
    #     "tune registry.")
    # required_named.add_argument(
    #     "--env", 
    #     type=str, 
    #     default="quadruped_env", 
    #     help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.")
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_true",
        help="Wrap environment in gym Monitor to record video. NOTE: This "
        "option is deprecated: Use `--video-dir [some dir]` instead.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Specifies the directory into which videos of all episode "
        "rollouts will be stored.")
    parser.add_argument(
        "--steps",
        default=1000,
        help="Number of timesteps to roll out (overwritten by --episodes).")
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out (overrides --steps).")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.")
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.")
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.")
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.")
    return parser


def run(args, parser):
    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")

    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    # Set num_workers to be at least 2.
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
        config["num_workers"] = 1

    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings.
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init(ignore_reinit_error=True)

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)
    # Load state from checkpoint.
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    # Determine the video output directory.
    # Deprecated way: Use (--out|~/ray_results) + "/monitor" as dir.
    video_dir = None
    if args.monitor:
        video_dir = os.path.join(
            os.path.dirname(args.out or "")
            or os.path.expanduser("~/ray_results/"), "monitor")
    # New way: Allow user to specify a video output path.
    elif args.video_dir:
        video_dir = os.path.expanduser(args.video_dir)

    # Do the actual rollout.
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        rollout(agent, args.env, num_steps, num_episodes, saver,
                args.no_render, video_dir)
    agent.stop()


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=None,
            no_render=True,
            video_dir=None):
    policy_agent_mapping = default_policy_agent_mapping

    if saver is None:
        saver = RolloutSaver()

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        print('we here')
        sys.exit()
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        print('actually here')
        sys.exit()
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym.wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda x: True,
            force=True)

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            # if not no_render:
            #     env.render()
            time.sleep(0.01)
            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1


if __name__ == "__main__":
    parser = create_parser()
    
    parser.add_argument("--run", default=alg)
    parser.add_argument("--env", default="quadruped_env")
    parser.add_argument("--checkpoint", default=checkpoint)
    print(parser)
    args = parser.parse_args()

    # Old option: monitor, use video-dir instead.
    if args.monitor:
        deprecation_warning("--monitor", "--video-dir=[some dir]")
    # User tries to record videos, but no-render is set: Error.
    if (args.monitor or args.video_dir) and args.no_render:
        raise ValueError(
            "You have --no-render set, but are trying to record rollout videos"
            " (via options --video-dir/--monitor)! "
            "Either unset --no-render or do not use --video-dir/--monitor.")
    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!")
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!")

    run(args, parser)
