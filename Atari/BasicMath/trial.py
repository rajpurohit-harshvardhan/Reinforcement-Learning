import cv2
import pickle
import ale_py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# gym.register_envs(ale_py)

# Initialize the environment
env = gym.make("ALE/BasicMath-v5")


# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor

total_episodes = 1000  # Number of episodes
max_steps = 10000

epsilon = 1
decay_rate = 1 / total_episodes

screen_width = 160  # Width of the screen

player_pos = np.linspace(0, 9, 10)
player_velocity = np.linspace(2, 11, 10)
can_shoot = np.linspace(0, 1, 2)
distance_to_left_edge = np.linspace(0, 100, 10)
distance_to_right_edge = np.linspace(0, 100, 10)

print("Observation space ::", tuple([player_pos, player_velocity, can_shoot, distance_to_left_edge, distance_to_right_edge]))
