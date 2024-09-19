import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt

alpha = 0.09
epsilon = 0.5
gamma = 0.99

total_episodes = 10000
max_steps = 1000

velocity_space = np.linspace(-4, 4, 22)
angle_space = np.linspace(-12, 12, 22)
position_space = np.linspace(-4, 4, 10)


def create_environment():
    env = gym.make('CartPole-v1')
    Q = np.zeros((len(angle_space)+1, len(velocity_space)+1, len(position_space)+1, env.action_space.n))
    return env, Q


def choose_action(state, Q, env):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action


def update(state, state2, reward, action, action2, Q):
    predict = Q[state][action]
    target = reward + gamma * Q[state2][action2]
    Q[state][action] = Q[state][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q):
    data = {}
    episodes_beating_game = 0
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state_a = np.digitize(math.degrees(state[2]), angle_space)
        state_v = np.digitize(state[3], velocity_space)
        state_p = np.digitize(state[0], position_space)
        state1 = (state_a, state_v, state_p)

        action1 = choose_action(state1, Q, env)
        print("Episode ::", episode)
        while t < max_steps:
            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)
            newstate_a = np.digitize(math.degrees(state2[2]), angle_space)
            newstate_v = np.digitize(state2[3], velocity_space)
            newstate_p = np.digitize(state2[0], position_space)
            state2 = (newstate_a, newstate_v, newstate_p)


            # Choosing the next action
            action2 = choose_action(state2, Q, env)

            # Learning the Q-value
            Q = update(state1, state2, reward, action1, action2, Q)

            state1 = state2
            action1 = action2

            # Updating the respective values
            t += 1

            # If at the end of learning process
            if done:
                data[episode] = t
                if t > 500:
                    episodes_beating_game += 1
                break

    lists = sorted(data.items())  # sorted by key, return a list of tuples
    lists2 = sorted(data.values())  # sorted by values, return a list of values
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.show()
    plt.plot(lists2)
    plt.show()

    ## PLOTTING BOTH THE GRAPHS IN THE SAME IMAGE
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.plot(lists2)
    # plt.show()

    print("Total Episodes where the agent aced the game:", episodes_beating_game)
    return Q


def play(Q):
    env = gym.make('CartPole-v1', render_mode='human')
    for i in range(5):
        t = 0
        reset_state = env.reset()[0]
        state_a = np.digitize(math.degrees(reset_state[2]), angle_space)
        state_v = np.digitize(reset_state[3], velocity_space)
        state_p = np.digitize(reset_state[0], position_space)
        state = (state_a, state_v, state_p)
        while t < max_steps:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state]))
            cart_pos = state2[1]
            newstate_a = np.digitize(math.degrees(state2[2]), angle_space)
            newstate_v = np.digitize(state2[3], velocity_space)
            newstate_p = np.digitize(state2[0], position_space)
            state = (newstate_a, newstate_v, newstate_p)

            t += 1

            if done:
                break
                if state < 1 or state > 21:
                        print("BREAKING DUE TO DEGREES ::", state_degrees)
                        break
        print("STEPS ::: ", t)


def main():
    env,Q = create_environment()
    # print(env, Q)
    Q = train(total_episodes, max_steps, env, Q)
    play(Q)

main()