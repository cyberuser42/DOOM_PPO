from vizdoom import * 
from vizdoom import GameVariable

import cv2
import random
import time 
import numpy as np

from gym import Env 
from gym.spaces import Discrete, Box

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from available_actions import get_available_actions

class VizDoomGym(Env): 
    # Function that is called when we start the env
    def __init__(self, render=False, config='./scenarios/deadly_corridor_s1.cfg'): 
        # Inherit from Env
        super().__init__()
        # Setup the game 
        self.game = DoomGame()
        self.game.load_config(config)
        
        # Render frame logic
        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        # Start the game 
        self.game.init()

        self.possible_actions = get_available_actions(np.array(self.game.get_available_buttons()))
        self.action_space = Discrete(len(self.possible_actions))

        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(7)
        
        # Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.damagecount = 0
        self.ammo = 52 ## CHANGED
        #self.x_coord = 0
        
        
    # This is how we take a step in the environment
    def step(self, action):
        movement_reward = self.game.make_action(self.possible_actions[action], 2) 
        
        reward = 0 
        # Get all the other stuff we need to retun 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            
            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, damagecount, ammo = game_variables
            
            #x_coord = self.game.get_game_variable(GameVariable.POSITION_X)

            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken

            damagecount_delta = damagecount - self.damagecount
            self.damagecount = damagecount

            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            #x_coord_delta = x_coord - self.x_coord
            #self.x_coord = x_coord

            reward = (movement_reward + damage_taken_delta*10 + damagecount_delta*200  + ammo_delta*5)
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
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
