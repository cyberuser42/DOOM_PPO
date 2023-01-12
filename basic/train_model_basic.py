from vizdoom import * 

import os 
import cv2
import random
import time 
import numpy as np

from gym import Env 
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


CHECKPOINT_DIR = 'train_basic/'
LOG_DIR = 'log_basic/'

# Create Vizdoom OpenAI Gym Environment
class VizDoomGym(Env): 
    # Function that is called when we start the env
    def __init__(self, render=False, config='scenarios/basic.cfg'):
        # Inherit from Env
        super().__init__()
        # Setup the game 
        self.game = DoomGame()
        self.game.load_config(config)
        self.game.set_window_visible(False)

        # Render frame logic
       #if render == False: 
        #    self.game.set_window_visible(False)
        #else:
        #    self.game.set_window_visible(True)
        
        # Start the game 
        self.game.init()
        
        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(self.game.get_available_buttons_size())
        self.possible_actions = np.eye(self.action_space.n).tolist()
    
    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step 
        reward = self.game.make_action(self.possible_actions[action], 4) 
        
        # Get all the other stuff we need to retun 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        done = self.game.is_episode_finished()
        info = {"info":info}

        return state, reward, done, info
    
    # Define how to render the game or environment 
    def render(): 
        pass
    
    # What happens when we start a new game 
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    # Grayscale the game frame and resize it 
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    # Call to close down the game
    def close(self): 
        self.game.close()


def create_env(scenario: str, **kwargs) -> VizDoomGym:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.set_window_visible(False)
    game.init()

    # Wrap the game with the Gym adapter.
    return VizDoomGym(game, **kwargs)

def create_vec_env(n_envs: int = 1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env(**kwargs)] * n_envs))

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'model{}k'.format(self.n_calls/10000))
            self.model.save(model_path)

        return True

training_env, eval_env = create_vec_env(n_envs=8, scenario="basic"), create_vec_env(n_envs=1, scenario="basic")

# Non rendered environment
#env = VizDoomGym()

def create_agent(env, **kwargs):
    return PPO(policy="CnnPolicy",
                   env=env,
                   n_steps=4096,
                   batch_size=32,
                   learning_rate=1e-4,
                   tensorboard_log='logs/tensorboard',
                   verbose=1,
                   seed=0,
                   **kwargs)


model = create_agent(training_env)

# Define an evaluation callback that will save the model when a new reward record is reached.
evaluation_callback = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=5000, log_path='logs/evaluations/basic', best_model_save_path='logs/models/basic')


model.learn(total_timesteps=100000, callback=evaluation_callback)

training_env.close()
eval_env.close()

