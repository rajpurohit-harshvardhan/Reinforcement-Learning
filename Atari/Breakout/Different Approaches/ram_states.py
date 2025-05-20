import gymnasium as gym
import ale_py

# Initialize the Breakout environment with RAM observation
env = gym.make('ALE/Breakout-ram-v5', render_mode='human')

# Reset the environment to start
observation, info = env.reset()

ram_state = env.unwrapped.ale.getRAM()
paddle_position = ram_state[72]
ball_position_x = ram_state[99]
ball_position_y = ram_state[101]
previous_life = ram_state[57]
previous_score = ram_state[77]

# 'observation' now contains the RAM state as a 128-element array
print("Initial RAM State:", ram_state)
while True:
    # Example: Take a random action
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)

    ram_state = env.unwrapped.ale.getRAM()
    padding_position = ram_state[72]
    ball_position_x = ram_state[99]
    ball_position_y = ram_state[101]
    life = ram_state[57]
    score = ram_state[77]

    # if life != previous_life:
    if life != previous_life:
        print("Updated RAM State:", ram_state)
        previous_life = life

    # 'observation' contains the updated RAM state
    # print("Updated RAM State:", observation)

# Close the environment
env.close()
