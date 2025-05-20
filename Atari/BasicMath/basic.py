import gymnasium as gym
import ale_py

# Initialize the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")  # Set render_mode to 'human' for visualization

# Reset the environment
obs, info = env.reset()

# Run a loop for random actions
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Check if the episode is done
    done = terminated or truncated

    # Print step information
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

# Close the environment
env.close()
