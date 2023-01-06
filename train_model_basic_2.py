import cv2
import matplotlib.pyplot as plt
import numpy as np
import vizdoom
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import evaluation
from stable_baselines3.common import policies

from common import envs

def create_env(**kwargs) -> envs.DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config('scenarios/basic.cfg')
    game.init()

    # Wrap the environment with the Gym adapter.
    return envs.DoomEnv(game, **kwargs)