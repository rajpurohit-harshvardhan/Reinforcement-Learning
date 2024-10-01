import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

alpha = 0.99  # Learning Rate
gamma = 0.99  # Discount Rate

total_episodes = 10000
max_steps = 200

epsilon = 0
decay_rate = 2 / total_episodes

velocity_space = np.linspace(-0.07, 0.07, 20)
position_space = np.linspace(-1.2, 0.6, 20)


def create_environment():
    env = gym.make('MountainCar-v0')
    Q = np.zeros((len(position_space), len(velocity_space), env.action_space.n))
    return env, Q


def choose_action(state_p, state_v, Q, env):
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state_p, state_v, :])
    return action


def update(state1_p, state1_v, state2_p, state2_v, reward, action, action2, Q):
    predict = Q[state1_p, state1_v][action]
    target = reward + gamma * Q[state2_p, state2_v][action2]
    Q[state1_p, state1_v][action] = Q[state1_p, state1_v][action] + alpha * (target - predict)
    return Q


def train(total_episodes, max_steps, env, Q, epsilon):
    data = {}
    for episode in range(total_episodes):
        data[episode] = 0
        t = 0
        state = env.reset()[0]
        state_p = np.digitize(state[0], position_space)
        state_v = np.digitize(state[1], velocity_space)
        # state1 = (state_p, state_v)

        action1 = choose_action(state_p, state_v, Q, env)
        print("Episode ::", episode)
        rewards = 0

        while rewards > -1000:
            state2, reward, done, trunc, info = env.step(action1)

            reward = int(state2[0]*10)/10  # new_method

            # Condition based method
            # if state2[0] >= 0.6:
            #     reward = 1
            # elif state2[0] >= 0.4:
            #     reward = 0.3
            # elif state2[0] >= 0.3:
            #     reward = -0.1
            # elif state2[0] >= 0:
            #     reward = -0.5
            # elif state2[0] >= -0.3:
            #     reward = -0.9
            # else:
            #     reward = -1

            newstate_p = np.digitize(state2[0], position_space)
            newstate_v = np.digitize(state2[1], velocity_space)
            # state2 = (newstate_p, newstate_v)

            action2 = choose_action(newstate_p, newstate_v, Q, env)

            Q = update(state_p, state_v, newstate_p, newstate_v, reward, action1, action2, Q)

            state_p = newstate_p
            state_v = newstate_v
            action1 = action2

            t += 1
            rewards += reward

            if done:
                data[episode] = rewards
                break

        epsilon = max(epsilon - decay_rate, 0)

    # saving the Q table in a file
    f = open("mountain_car_all_conditions.pkl", "wb")
    pickle.dump(Q, f)
    f.close()

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
    env = gym.make('MountainCar-v0', render_mode='human')

    # Read the Q table from the File
    f = open("mountain_car_all_conditions.pkl", "rb")
    Q = pickle.load(f)
    f.close()

    for i in range(5):
        t = 0
        reset_state = env.reset()[0]
        print("Starting at position: ", reset_state[0])
        state_p = np.digitize(reset_state[0], position_space)
        state_v = np.digitize(reset_state[1], velocity_space)
        # state = (state_p, state_v)

        rewards = 0
        while rewards > -1000:
            state2, reward, done, trunc, info = env.step(np.argmax(Q[state_p, state_v]))
            newstate_p = np.digitize(state2[0], position_space)
            newstate_v = np.digitize(state2[1], velocity_space)
            # state = (newstate_p, newstate_v)

            state_p = newstate_p
            state_v = newstate_v

            t += 1
            rewards += reward
            if done:
                break

        print("STEPS ::: ", t)


def main():
    env,Q = create_environment()
    Q = train(total_episodes, max_steps, env, Q, epsilon)
    play(Q)

main()