# Learning
This is the master branch for learning-related activities for quadrupeds.

## Installation

Recommend using a virtualenv with python3.6 (tested). Recent pybullet, numpy, and tensorflow==1.14.0, stable-baselines, rllib, etc. 
Then, install usc_learning as a package:

`pip install -e . `

After sourcing ROS path, you can also run examples in [ROS](./usc_learning/ros_interface).

## Code structure

- [learning](./usc_learning/learning) for rllib and stable-baselines extensions and modifications
- [imitation_tasks](./usc_learning/imitation_tasks) for any trajectories from the optimization, and classes to compare current robot state with desired ones, etc.
- [envs](./usc_learning/envs) any robot classes and basic gym environments. See in particular [quadruped_master](./usc_learning/envs/quadruped_master) for a general interface for aliengo and laikago.
- [ROS](./usc_learning/ros_interface) call MPC in ROS for use in pybullet

## TODO

- Fix train_LSTM.py, which currently show error during training

## Resources
- [Google-motion-imitation](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf) can be a helpful resource. 
- [Imitation task specifically](https://github.com/google-research/motion_imitation/blob/master/motion_imitation/envs/env_wrappers/imitation_task.py)
