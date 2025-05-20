import gymnasium as gym
import pygame
import numpy as np
from ale_py import ALEInterface

# Initialize the Breakout environment
env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array", frameskip=2)
env.reset()

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((600, 400))  # Window for visualization
pygame.display.set_caption("Breakout Game")

# Actions mapping
ACTIONS = {
    "NOOP": 0,  # Do nothing
    "FIRE": 1,  # Start the game (press to begin)
    "RIGHT": 2,  # Move paddle right
    "LEFT": 3,  # Move paddle left
}

# Initialize variables
done = False
clock = pygame.time.Clock()
current_action = ACTIONS["NOOP"]


def get_key_action():
    """Detects key presses and returns the corresponding action."""
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return ACTIONS["LEFT"]
    elif keys[pygame.K_RIGHT]:
        return ACTIONS["RIGHT"]
    elif keys[pygame.K_SPACE]:
        return ACTIONS["FIRE"]
    else:
        return ACTIONS["NOOP"]


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Get action from keyboard
    current_action = get_key_action()

    # Step in the environment
    obs, reward, terminated, truncated, _ = env.step(current_action)
    done = terminated or truncated

    # Convert environment image to pygame display
    # frame = np.rot90(obs)  # Remove this line (causes unwanted rotation)
    frame = pygame.surfarray.make_surface(obs)
    frame = pygame.transform.rotate(frame, 270)
    frame = pygame.transform.flip(frame, True, False)  # Flip the screen vertically
    frame = pygame.transform.scale(frame, (600, 400))
    screen.blit(frame, (0, 0))

    pygame.display.flip()
    clock.tick(30)  # 30 FPS limit

# Cleanup
env.close()
pygame.quit()
