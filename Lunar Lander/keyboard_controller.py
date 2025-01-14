import gymnasium as gym
import pygame
from pygame.locals import *

# Initialize the environment
env = gym.make('LunarLander-v2', render_mode='human')
env.reset()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption('LunarLander-v2 Keyboard Control')

# Action mapping
action = 0  # Default to 'do nothing'
action_mapping = {
    K_LEFT: 1,   # Fire left orientation engine
    K_UP: 2,     # Fire main engine
    K_RIGHT: 3   # Fire right orientation engine
}

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key in action_mapping:
                action = action_mapping[event.key]
        elif event.type == KEYUP:
            if event.key in action_mapping:
                action = 0  # Reset to 'do nothing' when key is released

    # Take action in the environment
    observation, reward, done, truncated, info = env.step(action)
    env.render()

    if done:
        env.reset()

    clock.tick(30)  # Limit to 30 frames per second

env.close()
pygame.quit()
