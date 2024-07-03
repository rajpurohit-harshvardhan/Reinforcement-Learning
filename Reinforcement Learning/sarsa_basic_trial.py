## KINDLY IGNORE THIS FILE
# Just trying out different stuff
#

import gym
import numpy as np
from sarsa_custom_trial import CardPickingEnv

epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

env = CardPickingEnv(gym)
Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + alpha * (target - predict)


reward = 0

for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)

    while t < max_steps:
        # Visualizing the training
        env.render()

        # Getting the next state
        state2, reward, done, info = env.step(action1)

        # Choosing the next action
        action2 = choose_action(state2)

        # Learning the Q-value
        update(state1, state2, reward, action1, action2)

        state1 = state2
        action1 = action2

        # Updating the respective vaLues
        t += 1
        reward += 1

        # If at the end of learning process
        if done:
            break

print("Performance : ", reward / total_episodes)

# Visualizing the Q-matrix
print(Q)
