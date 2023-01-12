from vizdoom import * 

import cv2
import random
import time 
import numpy as np
import gym
from gym import Env 
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import os 
from stable_baselines3.common.callbacks import BaseCallback

from env import VizDoomGym
from TrainAndLoggingCallback import TrainAndLoggingCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv


CHECKPOINT_DIR = 'train_deadly_corridor_final/'
LOG_DIR = 'log_deadly_corridor_final/'

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

#def create_vec_env(**kwargs) -> VecTransposeImage:
#    vec_env = DummyVecEnv([lambda: VizDoomGym(**kwargs)])
#    return VecTransposeImage(vec_env)

#envs = [gym.make(VizDoomGym) for i in range(8)]
#vec_env = SubprocVecEnv(envs)

#vec_env = DummyVecEnv([lambda: VizDoomGym(config='deadly_corridor/scenarios/deadly_corridor_s1.cfg', render=False)])
#env = VecTransposeImage(vec_env)

# Non rendered environment
env = VizDoomGym(config='scenarios/deadly_corridor_s1.cfg', render=False)

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=0.1, gamma=0.95, gae_lambda=0.9)

model.learn(total_timesteps=500000, callback=callback)

env = VizDoomGym(config='scenarios/deadly_corridor_s2.cfg', render=False)
model.set_env(env)
model.learn(total_timesteps=400000, callback=callback)

env = VizDoomGym(config='scenarios/deadly_corridor_s3.cfg', render=False)
model.set_env(env)
model.learn(total_timesteps=400000, callback=callback)

env = VizDoomGym(config='scenarios/deadly_corridor_s4.cfg', render=False)
model.set_env(env)
model.learn(total_timesteps=400000, callback=callback)

env = VizDoomGym(config='scenarios/deadly_corridor_s5.cfg', render=False)
model.set_env(env)
model.learn(total_timesteps=400000, callback=callback)
