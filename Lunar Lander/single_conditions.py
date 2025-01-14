import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount Rate

total_episodes = 1000
max_steps = 30000

epsilon = 1
decay_rate = 2 / total_episodes

x_space = np.linspace(-0.5, 0.5, 10)
y_space = np.linspace(0, 1.5, 10)
x_velocity = np.linspace(-1, 1, 10)
y_velocity = np.linspace(-0.5, 0.5, 21)
angle_space = np.linspace(-0.1, 0.1, 22)
velocity_space = np.linspace(-0.2, 0.2, 21)
left_leg_bool = np.linspace(0, 1, 2)
right_leg_bool = np.linspace(0, 1, 2)


def create_environment():
    env = gym.make('LunarLander-v2')

    # Q = np.zeros((len(velocity_space)+1, len(y_velocity)+1, len(x_velocity)+1, env.action_space.n))
    Q = np.zeros((len(y_velocity)+1, len(velocity_space)+1, env.action_space.n))
    return env, Q


def choose_action_conditions(state):
    action = 2
    if state[5] > 0.04:
        action = 3
    elif state[3] > -0.35:
        action = 0
    elif state[5] > -0.22:
        action = 0
    elif state[4] > -0.04:
        action = 1
    elif state[6] > 0.5 or state[7] > 0.5:
        action = 2
    elif state[2] > 0.32:
        action = 2
    elif state[1] > -0.11:
        action = 1
    elif state[4] > 0.15:
        action = 1
    elif state[0] > -0.34:
        action = 1
    else:
        action = 2
    return action


def choose_actions(state):
    action = 2

    # angular_velocity

    if state[4] < -0.01:
        action = 3
    elif state[4] > 0.01:
        action = 1
    else:
        action = 2

    # if state[3] < -0.1:
    #     action = 2
    # elif state[3] > 0:
    #     action = 0
    # else:
    #
    #     if -0.05 < state[5] < 0.05:
    #         action = 2  # Solely based on Angular velocity the lander remains upright.
    #
    #         # if state[1] < 0.05:
    #         #     action = 2
    #         # else:
    #         #     action = 0
    #
    #         # print(state[3])
    #         # Y velocity
    #         # if state[3] < -0.5:
    #         #     action = 2
    #         # elif state[3] > 0.4:
    #         #     action = 0
    #         # else:
    #         #     # if state[0] < 0:
    #         #     #     action = 1
    #         #     # elif state[0] > 0:
    #         #     #     action = 3
    #         #     # else:
    #         #     #     print("^^^^^^^^^^^^^")
    #         #     #     action = 0
    #         #
    #         #     # X velocity
    #         #     # if state[2] < -0.1:
    #         #     #     action = 3
    #         #     # elif state[2] > 0.1:
    #         #     #     action = 1
    #         #     # else:
    #         #     #     action = 0
    #         #     action = 0
    #     elif state[5] > 0.05:
    #         action = 1
    #     elif state[5] < -0.05:
    #         action = 3
    #     else:
    #         print("###############################################################")
    #
    #
    # # Y position
    # # if state[1] < 0.05:
    # #     action = 2
    # # else:
    # #     action = 0
    #
    # # Y velocity
    # # if state[3] < -0.35:
    # #     action = 2
    # # elif state[3] > 0.2:
    # #     action = 0
    #
    # # X velocity
    # # if state[2] < -0.1:
    # #     action = 3
    # # elif state[2] > 0.1:
    # #     action = 1
    # # else:
    # #     action = 0
    #
    # # angle
    # # if state[4] < 0:
    # #     action = 1
    # # elif state[4] > 0:
    # #     action = 3
    # # else:
    # #     action = 2
    #
    # # X position
    # # if state[0] < 0:
    # #     # if state[1] < 0.05:
    # #     #     action = 2
    # #     # else:
    # #     action = 1
    # # elif state[0] > 0:
    # #     # if state[1] < 0.05:
    # #     #     action = 2
    # #     # else:
    # #     action = 3
    # # else:
    # #     # if state[1] < 0.05:
    # #     #     action = 2
    # #     # else:
    # #     action = 0
    return action


def choose_action(state, Q, env):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action


def update(state1, state2, reward, action, action2, Q):
    predict = Q[state1][action]
    target = reward + gamma * Q[state2][action2]
    Q[state1][action] = Q[state1][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state_x = np.digitize(state[0], x_space)
        state_y = np.digitize(state[1], y_space)
        state_x_velocity = np.digitize(state[2], x_velocity)
        state_y_velocity = np.digitize(state[3], y_velocity)
        state_angle = np.digitize(state[4], angle_space)
        state_angle_velocity = np.digitize(state[5], velocity_space)
        state_left_leg = np.digitize(state[6], left_leg_bool, True)
        state_right_leg = np.digitize(state[7], right_leg_bool, True)

        # state1 = (state_angle_velocity, state_y_velocity, state_x_velocity)
        state1 = (state_y_velocity)

        # action1 = choose_action_conditions(state)
        action1 = choose_actions(state)

        if episode % 1000 == 0:
            print("#### Episode:", episode, " :: action :", action1)

        # print("Episode ::", episode)
        rewards = 0
        calculated_reward = 0
        actions = {"0": 0, "1": 0, "2": 0, "3": 0}
        actions[str(action1)]+=1
        episode_finished = False

        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)

            new_state_x = np.digitize(state2[0], x_space)
            new_state_y = np.digitize(state2[1], y_space)
            new_state_x_velocity = np.digitize(state2[2], x_velocity)
            new_state_y_velocity = np.digitize(state2[3], y_velocity)
            new_state_angle = np.digitize(state2[4], angle_space)
            new_state_angle_velocity = np.digitize(state2[5], velocity_space)
            new_state_left_leg = np.digitize(state2[6], left_leg_bool, True)
            new_state_right_leg = np.digitize(state2[7], right_leg_bool, True)

            # new_state2 = (new_state_angle_velocity, new_state_y_velocity, new_state_x_velocity)
            new_state2 = (new_state_y_velocity)

            if episode < 100:
                action2 = 2
            else:
                action2 = choose_actions(state2)

            actions[str(action2)] += 1

            # assigning new reward
            if state[3] >= 0:
                reward = state[3] - abs(state[4])

            Q = update(state1, new_state2, reward, action1, action2, Q)

            action1 = action2

            t += 1
            rewards += reward
            calculated_reward += reward

            # if (done and calculated_reward > 100) or t > max_steps:
            if done:
                episode_finished = True
                data[episode] = rewards
                # print("Calculated_reward = ", calculated_reward, "Max steps:", t, "Actions Taken::",actions)
                calculated_reward = 0
                actions = {"0": 0, "1": 0, "2": 0, "3": 0}
                break

        epsilon = max(epsilon - decay_rate, 0)
        # print("Episode ::", episode, " - Rewards:", data[episode])

    # saving the Q table in a file
    f = open("lunar_lander_single.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

    # lists = sorted(data.items())  # sorted by key, return a list of tuples
    # lists2 = sorted(data.values())  # sorted by values, return a list of values
    # x, y = zip(*lists)  # unpack a list of pairs into two tuples
    # plt.plot(x, y)
    # plt.show()
    # plt.plot(lists2)
    # plt.show()

    ## PLOTTING BOTH THE GRAPHS IN THE SAME IMAGE
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.plot(lists2)
    # plt.show()

    return Q


def play(Q):
    env = gym.make('LunarLander-v2', render_mode='human')

    print("environment loaded")
    # Read the Q table from the File
    f = open("Backups\lunar_lander_conditions_1.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    print("Starting Playing the game.")
    for i in range(30):
        t = 0
        reset_state = env.reset()[0]
        print("Starting at position: ", reset_state[1])
        state_x = np.digitize(reset_state[0], x_space)
        state_y = np.digitize(reset_state[1], y_space)
        state_x_velocity = np.digitize(reset_state[2], x_velocity)
        state_y_velocity = np.digitize(reset_state[3], y_velocity)
        state_angle = np.digitize(reset_state[4], angle_space)
        state_angle_velocity = np.digitize(reset_state[5], velocity_space)
        state_left_leg = np.digitize(reset_state[6], left_leg_bool, True)
        state_right_leg = np.digitize(reset_state[7], right_leg_bool, True)

        # state = (state_angle_velocity, state_y_velocity, state_x_velocity)
        state = (state_y_velocity, state_angle_velocity)
        # print("Initial Y velocity: ", reset_state[3])

        action1 = np.argmax(Q[state])

        rewards = 0
        episode_finished = False
        while not episode_finished:
            state2, reward, done, trunc, info = env.step(action1)
            new_state_x = np.digitize(state2[0], x_space)
            new_state_y = np.digitize(state2[1], y_space)
            new_state_x_velocity = np.digitize(state2[2], x_velocity)
            new_state_y_velocity = np.digitize(state2[3], y_velocity)
            new_state_angle = np.digitize(state2[4], angle_space)
            new_state_angle_velocity = np.digitize(state2[5], velocity_space)
            new_state_left_leg = np.digitize(state2[6], left_leg_bool, True)
            new_state_right_leg = np.digitize(state2[7], right_leg_bool, True)

            # new_state = (new_state_angle_velocity, new_state_y_velocity, new_state_x_velocity)
            new_state = (new_state_y_velocity, new_state_angle_velocity)
            # print("Successive Y velocity: ", state2[3])

            action2 = np.argmax(Q[new_state])
            action1 = action2

            t += 1
            rewards += reward
            if done:
                episode_finished = True
                break

        print("STEPS ::: ", t)


def main():
    env, Q = create_environment()
    # Q = train(total_episodes, max_steps, env, Q, epsilon)
    # print(Q)
    play(Q)


main()
