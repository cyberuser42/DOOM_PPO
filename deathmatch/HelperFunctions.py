import cv2
import random
import time 
import numpy as np

from gym import Env 
from gym.spaces import Discrete, Box

from vizdoom import * 
from vizdoom.vizdoom import GameVariable

def shape_rewards(self, initial_reward: float):
    frag_reward = self.compute_frag_reward()
    damage_reward = self.compute_damage_reward()
    ammo_reward = self.compute_ammo_reward()
    health_reward = self.compute_health_reward()
    armor_reward = self.compute_armor_reward()
    distance_reward = self.compute_distance_reward(*self.player_position())

    return initial_reward + frag_reward + damage_reward + ammo_reward + health_reward + armor_reward + distance_reward

def compute_distance_reward(self, x, y):
    dx = self.last_x - x
    dy = self.last_y - y

    self.last_x = x
    self.last_y = y

    distance = np.sqrt(dx ** 2 + dy ** 2)
    d = distance - self.reward_threshold_distance

    if d > 0:
        reward = 0.0005 #self.reward_factor_distance * d
    else:
        reward = -0.0005 #self.penalty_factor_distance * d

    self.log_reward_stat('distance', reward)

    return reward

def compute_frag_reward(self):
    frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
    reward = self.reward_factor_frag * (frags - self.last_frags)

    self.last_frags = frags
    self.log_reward_stat('frag', reward)

    return reward

def compute_damage_reward(self):
    damage_dealt = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
    reward = self.reward_factor_damage * (damage_dealt - self.last_damage_dealt)

    self.last_damage_dealt = damage_dealt
    self.log_reward_stat('damage', reward)

    return reward

def compute_health_reward(self):
    # When player is dead, the health game variable can be -999900
    health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)

    health_reward = self.reward_factor_health_increment * max(0, health - self.last_health)
    health_penalty = self.reward_factor_health_decrement * min(0, health - self.last_health)
    reward = health_reward - health_penalty

    self.last_health = health
    self.log_reward_stat('health', reward)

    return reward

def compute_armor_reward(self):
    armor = self.game.get_game_variable(GameVariable.ARMOR)
    reward = self.reward_factor_armor_increment * max(0, armor - self.last_armor)
    self.last_armor = armor
    self.log_reward_stat('armor', reward)

    return reward

def compute_ammo_reward(self):
    self.weapon_state = self.get_weapon_state()

    new_ammo_state = self.get_ammo_state()
    ammo_diffs = (new_ammo_state - self.ammo_state) * self.weapon_state
    ammo_reward = self.reward_factor_ammo_increment * max(0, np.sum(ammo_diffs))
    ammo_penalty = self.reward_factor_ammo_decrement * min(0, np.sum(ammo_diffs))
    reward = ammo_reward - ammo_penalty
    self.ammo_state = new_ammo_state
    self.log_reward_stat('ammo', reward)

    return reward

def player_position(self):
    return self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(
        GameVariable.POSITION_Y)

def get_ammo_state(self):
    ammo_state = np.zeros(10)

    for i in range(10):
        ammo_state[i] = self.game.get_game_variable(AMMO_VARIABLES[i])

    return ammo_state

def get_weapon_state(self):
    # ten unique weapons
    weapon_state = np.zeros(10)

    for i in range(10):
        weapon_state[i] = self.game.get_game_variable(WEAPON_VARIABLES[i])

    return weapon_state

def log_reward_stat(self, kind: str, reward: float):
    self.rewards_stats[kind] += reward

def reset_player(self):
    self.last_health = 100
    self.last_armor = 0
    self.game.respawn_player()
    self.last_x, self.last_y = self.player_position()
    self.ammo_state = self.get_ammo_state()

def auto_change_weapon(self):
    # Change weapons to one with ammo
    possible_weapons = np.flatnonzero(self.ammo_state * self.weapon_state)
    possible_weapon = possible_weapons[-1] if len(possible_weapons) > 0 else None

    current_selection = self.game.get_game_variable(GameVariable.SELECTED_WEAPON)
    new_selection = possible_weapon if possible_weapon != current_selection else None

    return new_selection

def respawn_if_dead(self):
    if not self.game.is_episode_finished():
        # Check if player is dead
        if self.game.is_player_dead():
            self.deaths += 1
            self.reset_player()

def print_state(self):
    server_state = self.game.get_server_state()
    player_scores = list(zip(
        server_state.players_names,
        server_state.players_frags,
        server_state.players_in_game))
    player_scores = sorted(player_scores, key=lambda tup: tup[1])

    print('*** DEATHMATCH RESULTS ***')
    for player_name, player_score, player_ingame in player_scores:
        if player_ingame:
            print(f' - {player_name}: {player_score}')

def grayscale(self, observation):
    gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
    state = np.reshape(resize, (100,160,1))
    return state
