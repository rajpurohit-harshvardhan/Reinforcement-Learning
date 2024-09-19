import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt

alpha = 0.082
epsilon = 0.01
gamma = 0.99

total_episodes = 30000
max_steps = 500


def get_epsilon(t):
    return max(0.1, min(1., 1. - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(0.1, min(1., 1. - math.log10((t + 1) / 25)))


# def return_state_index_based_on_ang_velocity(state):
#     if state < -5:
#         return 0
#     elif -5 <= state <= -4.5:
#         return 1
#     elif -4.5 <= state <= -4:
#         return 2
#     elif -4 <= state <= -3.5:
#         return 3
#     elif -3.5 <= state <= -3:
#         return 4
#     elif -3 <= state <= -2.5:
#         return 5
#     elif -2.5 <= state <= -2:
#         return 6
#     elif -2 <= state <= -1.5:
#         return 7
#     elif -1.5 <= state <= -1:
#         return 8
#     elif -1 <= state <= -0.5:
#         return 9
#     elif -0.5 <= state <= 0:
#         return 10
#     elif 0 <= state <= 0.5:
#         return 11
#     elif 0.5 <= state <= 1:
#         return 12
#     elif 1 <= state <= 1.5:
#         return 13
#     elif 1.5 <= state <= 2:
#         return 14
#     elif 2 <= state <= 2.5:
#         return 15
#     elif 2.5 <= state <= 3:
#         return 16
#     elif 3 <= state <= 3.5:
#         return 17
#     elif 3.5 <= state <= 4:
#         return 18
#     elif 4 <= state <= 4.5:
#         return 19
#     elif 4.5 <= state <= 5:
#         return 20
#     elif 5 < state:
#         return 21
def return_state_index_based_on_ang_velocity(state):
    if state < -3:
        return 0
    elif -3 <= state < -2.7:
        return 1
    elif -2.7 <= state < -2.4:
        return 2
    elif -2.4 <= state < -2.1:
        return 3
    elif -2.1 <= state < -1.8:
        return 4
    elif -1.8 <= state < -1.5:
        return 5
    elif -1.5 <= state < -1.2:
        return 6
    elif -1.2 <= state < -0.9:
        return 7
    elif -0.9 <= state < -0.6:
        return 8
    elif -0.6 <= state < -0.3:
        return 9
    elif -0.3 <= state < 0:
        return 10
    elif 0 <= state < 0.3:
        return 11
    elif 0.3 <= state < 0.6:
        return 12
    elif 0.6 <= state < 0.9:
        return 13
    elif 0.9 <= state < 1.2:
        return 14
    elif 1.2 <= state < 1.5:
        return 15
    elif 1.5 <= state < 1.8:
        return 16
    elif 1.8 <= state < 2.1:
        return 17
    elif 2.1 <= state < 2.4:
        return 18
    elif 2.4 <= state < 2.7:
        return 19
    elif 2.7 <= state < 3:
        return 20
    elif 3 < state:
        return 21


def return_state_index_based_on_angle(state):
    if -24 <= state <= -22:
        return 0
    elif -22 <= state <= -20:
        return 1
    elif -20 <= state <= -18:
        return 2
    elif -18 <= state <= -16:
        return 3
    elif -16 <= state <= -14:
        return 4
    elif -14 <= state <= -12:
        return 5
    elif -12 <= state <= -10:
        return 6
    elif -10 <= state <= -8:
        return 7
    elif -8 <= state <= -6:
        return 8
    elif -6 <= state <= -4:
        return 9
    elif -4 <= state <= -2:
        return 10
    elif -2 <= state <= 0:
        return 11
    elif 0 <= state <= 2:
        return 12
    elif 2 <= state <= 4:
        return 13
    elif 4 <= state <= 6:
        return 14
    elif 6 <= state <= 8:
        return 15
    elif 8 <= state <= 10:
        return 16
    elif 10 <= state <= 12:
        return 17
    elif 12 <= state <= 14:
        return 18
    elif 14 <= state <= 16:
        return 19
    elif 16 <= state <= 18:
        return 20
    elif 18 <= state <= 20:
        return 21
    elif 20 <= state <= 22:
        return 22
    elif 22 <= state <= 24:
        return 23


def return_state_index(angle, velocity):
    state_angle = return_state_index_based_on_angle(math.degrees(angle))
    state_velocity = return_state_index_based_on_ang_velocity(velocity)
    if state_angle == state_velocity:
        return state_angle
    else:
        return state_velocity


def create_environment():
    env = gym.make('CartPole-v0')
    Q = np.zeros((22, env.action_space.n))  # For 22 angular velocity parts
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
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state1 = return_state_index(state[2], state[3])  # for angular velocity

        action1 = choose_action(state1, Q, env)
        print("Episode ::", episode)
        while t < max_steps:
            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, done, trunc, info = env.step(action1)
            cart_pos = state2[1]
            state2 = return_state_index(state2[2], state2[3])

            if state2 is None:
                data[episode] = t
                break

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
                if state1 < 1 or state1 > 21:
                    if cart_pos < -1 or cart_pos > 1:
                        data[episode] = t
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

    return Q


def play(Q):
    env = gym.make('CartPole-v0', render_mode='human')
    for i in range(5):
        t = 0
        reset_state = env.reset()[0]
        state = return_state_index(reset_state[2], reset_state[3])  # for Angular velocity
        while t < 500:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state]))
            cart_pos = state2[1]
            state = return_state_index(state2[2], state2[3])  # for angular velocity

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